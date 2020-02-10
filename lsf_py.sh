#!/bin/sh
#BSUB -q normal
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -n 1
#BSUB -J "step1_test"
#BSUB -R span[ptile=1]
#BSUB -m "user-g4a60"
#BSUB -gpu num=2
# OMP_NUM_THREADS=1 python ./utils/kits19_dataloader_nnunet.py
python residual_vnet.py