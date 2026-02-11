#!/bin/bash
python main.py -p random -m resnet8_subset31_pretrained_orig -r --stop-at codegen
cd handcraft
python patch_inode.py resnet8_subset31_pretrained_orig_evl
cd ..
python main.py -p random -m resnet8_subset31_pretrained_orig -r