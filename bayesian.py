import os
import time
import random
import argparse

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

from config import cfg
from dataset import TrainDataset
from models import ModelBuilder, SegmentationModule
from models.models import C1DeepSup, PPMDeepsup
from utils import AverageMeter, setup_logger
from lib.nn import user_scattered_collate, async_copy_to


class BayesianC1(PyroModule):
    def __init__(self, c1_module, device=None):
        super().__init__()
        c1_module = c1_module.to(device)
        self.cbr = c1_module.cbr
        conv_last = c1_module.conv_last
        self.conv_last = PyroModule[nn.Conv2d](
            conv_last.in_channels, conv_last.out_channels, 1, 1, 0)
        self.conv_last.weight = PyroSample(dist.Normal(
            torch.zeros_like(conv_last.weight),
            torch.full_like(conv_last.weight, cfg.BAYESIAN.prior_scale)).to_event(2))
        self.conv_last.bias = PyroSample(dist.Normal(
            torch.zeros_like(conv_last.bias),
            torch.full_like(conv_last.bias, cfg.BAYESIAN.prior_scale)).to_event(1))

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

    def predict(self, feed_dict):
        conv_out = self.encoder(feed_dict['img_data'])[0]
        logits = self.predictive(conv_out)['_RETURN']
        return logits.mean(0).argmax(1)

    # Copied from models.models
    def _pixel_acc(self, preds, label):
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc.item()

    def accuracy(self, feed_dict):
        preds = self.predict(feed_dict)
        return self._pixel_acc(preds, feed_dict['seg_label'])


# train one epoch
def train(segmentation_module, iterator, history, epoch, cfg, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

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
        ave_total_loss.update(loss)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            acc = segmentation_module.accuracy(batch_data)
            ave_acc.update(acc*100)
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss)
            history['train']['acc'].append(acc)


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder) = nets

    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/bayesian_history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/bayesian_decoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    pyro.get_param_store().save('{}/pyro_epoch_{}.pth'.format(cfg.DIR, epoch))


def main(cfg, gpu):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder).cuda()
    net_decoder = BayesianC1(ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder).cuda())

    segmentation_module = BayesianSegmentationModule(
        net_encoder, net_decoder)

    # Dataset and Loader
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

    # create loader iterator
    iterator_train = iter(loader_train)

    nets = (net_encoder, net_decoder)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.BAYESIAN.start_epoch, cfg.BAYESIAN.num_epoch):
        train(segmentation_module, iterator_train, history, epoch+1, cfg, gpu)

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1)

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
    logger.info("Running with config:\n{}".format(cfg))

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

    main(cfg, args.gpu)
