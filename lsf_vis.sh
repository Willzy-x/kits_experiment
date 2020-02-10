#!/bin/sh
#BSUB -q normal
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -n 1
#BSUB -J "vis_step1"
#BSUB -R span[ptile=1]
#BSUB -m "node03"
#BSUB -gpu num=2
python visual.py --vis_op gt --case_id 199 --model_path ./models/fullseg_vnet4_ker3/ --model_name vnet_step1_165.tar
