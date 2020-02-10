#!/bin/sh
#BSUB -q normal
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -n 1
#BSUB -J "full_seg_aug_prob+0.5_dice_con"
#BSUB -R span[ptile=1]
#BSUB -m "user-g4a60"
#BSUB -gpu num=2
python main.py > full_seg_aug_prob+0.5_con.log
