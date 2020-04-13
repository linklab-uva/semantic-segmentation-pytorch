#!/usr/bin/env python3
import os, json
from PIL import Image

if __name__ == '__main__':
    tr_images = sorted(f.path for f in os.scandir('images/train/'))
    tr_labels = sorted(f.path for f in os.scandir('labels/train/'))
    val_images = sorted(f.path for f in os.scandir('images/val/'))
    val_labels = sorted(f.path for f in os.scandir('labels/val/'))

    with open('training.odgt', 'w') as f:
        for im, lb in zip(tr_images, tr_labels):
            i = Image.open(im)
            data = {'fpath_img': im, 'fpath_segm': lb, 'width': i.width, 'height': i.height}
            f.write(json.dumps(data)+'\n')

    with open('validation.odgt', 'w') as f:
        for im, lb in zip(val_images, val_labels):
            i = Image.open(im)
            data = {'fpath_img': im, 'fpath_segm': lb, 'width': i.width, 'height': i.height}
            f.write(json.dumps(data)+'\n')
