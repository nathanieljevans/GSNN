#!/bin/sh 

DATA=../../data/
OUT=../output/exp1/
PROC=$OUT/proc/

rm -r $OUT
mkdir $OUT 
mkdir $PROC

python make_data.py --data $DATA --out $PROC --pathways R-HSA-9006934 --feature_space landmark --targetome_targets #| tee $PROC/make_data.log

python train_gsnn.py --data $PROC --out $OUT --dropout 0.5 --channels 2 --lr 5e-3 --clip_grad 2
python train_nn.py --data $PROC --out $OUT --dropout 0.2 --channels 64 --lr 1e-3 --clip_grad 2
python train_gnn.py --data $PROC --out $OUT  --lr 5e-3 --clip_grad 2


#python train_gsnn.py --data ../output/exp1/proc/ --out ../output/exp1/ --dropout 0.2 --channels 4 --lr 5e-3 --clip_grad 2
# python train_nn.py --data ../output/exp1/proc/ --out ../output/exp1/ --dropout 0.2 --channels 64 --lr 1e-4 --clip_grad 2 --ignore_cuda
