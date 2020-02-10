#!/bin/sh
#BSUB -q normal
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -n 1
#BSUB -J "step1_test"
#BSUB -R span[ptile=1]
#BSUB -m "user-g4a60"
#BSUB -gpu num=2
python test_step2.py --test_op case --start_id 189 --end_id 209 \
--model_path "/home/b26170223/Documents/kits_ab/work/20200202_2006 full_seg_vnet_4_ker3_aug(prob+0.1)_dice_con" \
--model_name vnet_step1_965.tar \
--data_suffix npy --strides 40 40 20 \
--sliding_window true
