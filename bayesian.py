import os
import time
import random
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer import Predictive
from PIL import Image
from tqdm import tqdm

from config import cfg
from dataset import TrainDataset, ValDataset, TestDataset
from models import ModelBuilder, SegmentationModule
from models.models import C1DeepSup, PPMDeepsup
from utils import AverageMeter, setup_logger, accuracy, intersectionAndUnion, load_colors, load_names
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy

CAR_CLASS = 13

colors = None
names = None


class BayesianC1_1(PyroModule):
    def __init__(self, c1_module, device=None):
        super().__init__()
        c1_module = c1_module.to(device)
        self.cbr = c1_module.cbr
        conv_last = c1_module.conv_last
        self.conv_last = PyroModule[nn.Conv2d](
            conv_last.in_channels, conv_last.out_channels, 1, 1, 0)
        self.conv_last.weight = PyroSample(dist.Normal(
            torch.zeros_like(conv_last.weight),
            torch.full_like(conv_last.weight, cfg.BAYESIAN.prior_scale)
        ).to_event(2))
        self.conv_last.bias = PyroSample(dist.Normal(
            torch.zeros_like(conv_last.bias),
            torch.full_like(conv_last.bias, cfg.BAYESIAN.prior_scale)
        ).to_event(1))

    def forward(self, conv_out, y=None):
        x = self.cbr(conv_out)
        x = self.conv_last(x)
        x = nn.functional.log_softmax(x, dim=1)

        if y is not None:
            mask = y >= 0
            x_reshaped = x.permute(0, 2, 3, 1)[mask]
            y_reshaped = y[mask]
            with pyro.plate("data", x_reshaped.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=x_reshaped), obs=y_reshaped)

        return x


class BayesianC1_2(PyroModule):
    def __init__(self, c1_module, device=None):
        super().__init__()
        c1_module = c1_module.to(device)

        conv1 = c1_module.cbr[0]
        self.conv1 = PyroModule[nn.Conv2d](
            conv1.in_channels, conv1.out_channels, 3, 1, 1, bias=False)
        self.conv1.weight = PyroSample(dist.Normal(
            torch.zeros_like(conv1.weight),
            torch.full_like(conv1.weight, cfg.BAYESIAN.prior_scale)
        ).to_event(2))

        self.bn1 = c1_module.cbr[1]
        self.relu = c1_module.cbr[2]

        conv_last = c1_module.conv_last
        self.conv_last = PyroModule[nn.Conv2d](
            conv_last.in_channels, conv_last.out_channels, 1, 1, 0)
        self.conv_last.weight = PyroSample(dist.Normal(
            torch.zeros_like(conv_last.weight),
            torch.full_like(conv_last.weight, cfg.BAYESIAN.prior_scale)
        ).to_event(2))
        self.conv_last.bias = PyroSample(dist.Normal(
            torch.zeros_like(conv_last.bias),
            torch.full_like(conv_last.bias, cfg.BAYESIAN.prior_scale)
        ).to_event(1))

    def forward(self, conv_out, y=None):
        x = self.conv1(conv_out)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_last(x)
        x = nn.functional.log_softmax(x, dim=1)

        if y is not None:
            mask = y >= 0
            x_reshaped = x.permute(0, 2, 3, 1)[mask]
            y_reshaped = y[mask]
            with pyro.plate("data", x_reshaped.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=x_reshaped), obs=y_reshaped)

        return x


class BayesianSegmentationModule(nn.Module):
    def __init__(self, net_enc, net_dec):
        super().__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.decoder_guide = AutoDiagonalNormal(poutine.block(self.decoder, hide=["obs"]))
        self.optim = Adam({"lr": cfg.BAYESIAN.lr})
        self.svi = SVI(self.decoder, self.decoder_guide, self.optim, loss=Trace_ELBO())
        self.predictive = Predictive(self.decoder, guide=self.decoder_guide,
            num_samples=cfg.BAYESIAN.predict_samples, return_sites=['_RETURN'])

        # Freeze encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    # Keep encoder in eval mode
    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()

    def train_batch(self, feed_dict):
        conv_out = self.encoder(feed_dict['img_data'])[0]
        loss = self.svi.step(conv_out, feed_dict['seg_label'])
        return loss

    def all_predictions(self, feed_dict, segSize=None):
        conv_out = self.encoder(feed_dict['img_data'])[0]
        preds = self.predictive(conv_out)['_RETURN']
        if segSize is not None:
            preds = nn.functional.interpolate(
                preds.view(preds.shape[0]*preds.shape[1], *preds.shape[2:]),
                size=segSize, mode='bilinear', align_corners=False
                ).view(*preds.shape[:3], *segSize)
        return preds

    def predict(self, feed_dict, segSize=None):
        logits = self.all_predictions(feed_dict)
        preds = logits.mean(0)
        if segSize is not None:
            preds = nn.functional.interpolate(
                preds, size=segSize, mode='bilinear', align_corners=False)
        return preds

    def segment(self, feed_dict, segSize=None):
        return self.predict(feed_dict, segSize).argmax(1)

    # Copied from models.models
    def _pixel_acc(self, preds, label):
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc.item()

    def accuracy(self, feed_dict):
        preds = self.segment(feed_dict)
        return self._pixel_acc(preds, feed_dict['seg_label'])

    def class_heatmap(self, feed_dict, cls=CAR_CLASS):
        logits = self.all_predictions(feed_dict)
        cls_probs = logits[:, :, cls].mean(0).exp()
        return cls_probs


# train one epoch
def train(segmentation_module, iterator, history, epoch, cfg, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_total_loss = AverageMeter()
    avg_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        batch_data = async_copy_to(batch_data, gpu)
        data_time.update(time.time() - tic)

        # forward pass
        loss = segmentation_module.train_batch(batch_data)

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        avg_total_loss.update(loss)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            acc = segmentation_module.accuracy(batch_data)
            avg_acc.update(acc)
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'Accuracy: {:4.2%}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          avg_acc.average(), avg_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss)
            history['train']['acc'].append(acc)


def evaluate(segmentation_module, loader, cfg, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp = segmentation_module.predict(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)

        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))


def checkpoint(net_decoder, history, cfg, epoch):
    print('Saving checkpoints...')
    torch.save(
        history,
        '{}/bayesian2_history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        net_decoder.state_dict(),
        '{}/bayesian2_decoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    pyro.get_param_store().save('{}/pyro2_epoch_{}.pth'.format(cfg.DIR, epoch))


def load_checkpoint(net_decoder, cfg, epoch):
    net_decoder.load_state_dict(
        torch.load('{}/bayesian2_decoder_epoch_{}.pth'.format(cfg.DIR, epoch)))
    pyro.get_param_store().load('{}/pyro2_epoch_{}.pth'.format(cfg.DIR, epoch))


def visualize_result(data, pred, dir_result, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx]]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def main(cfg, gpu, eval=False, interactive=False):
    global net_encoder, net_decoder, segmentation_module, \
        dataset_train, loader_train, iterator_train, history, epoch, \
        dataset_val, loader_val
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder).cuda()
    net_decoder = BayesianC1_2(ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder).cuda())

    segmentation_module = BayesianSegmentationModule(
        net_encoder, net_decoder)

    if eval or interactive:
        load_checkpoint(net_decoder, cfg, 10)

    # Dataset and Loader
    if not eval or interactive:
        dataset_train = TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET,
            batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
            num_workers=cfg.TRAIN.workers,
            drop_last=True,
            pin_memory=True)
        print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

        iterator_train = iter(loader_train)

    if eval or interactive:
        dataset_val = ValDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET)
        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.VAL.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

    if interactive:
        return

    if eval:
        evaluate(segmentation_module, loader_val, cfg, gpu)
        print('Evaluation Done!')
    else:
        # Main loop
        history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

        for epoch in range(cfg.BAYESIAN.start_epoch, cfg.BAYESIAN.num_epoch):
            train(segmentation_module, iterator_train, history, epoch+1, cfg, gpu)

            # checkpointing
            checkpoint(net_decoder, history, cfg, epoch+1)

        print('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training with Bayesian Weights"
    )
    parser.add_argument(
        "--cfg",
        default="config/config/ade20k-mobilenetv2dilated-c1_deepsup-bdd100k.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu to use"
    )
    parser.add_argument('-e', '--eval', action='store_true')
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    # logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.BAYESIAN.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.BAYESIAN.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    cfg.TRAIN.batch_size = cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

    # random.seed(cfg.TRAIN.seed)
    # torch.manual_seed(cfg.TRAIN.seed)

    colors = load_colors(cfg.DATASET.colors_file)
    names = load_names(cfg.DATASET.names_file)

    main(cfg, args.gpu, args.eval, args.interactive)
