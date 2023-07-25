#!/bin/sh 

DATA=../../data/
OUT=../output/exp2/
PROC=$OUT/proc/

rm -r $OUT
mkdir $OUT 
mkdir $PROC

python make_data.py --data $DATA --out $PROC --pathways R-HSA-5683057 --feature_space landmark --targetome_targets --test_prop 0.2 --val_prop 0.

python train_gsnn.py --data $PROC --out $OUT --dropout 0.25 --channels 3 --lr 5e-3 --clip_grad 2 
python train_gsnn.py --data $PROC --out $OUT --dropout 0.25 --channels 3 --lr 5e-3 --clip_grad 2 --randomize
python train_nn.py --data $PROC --out $OUT --dropout 0.2 --channels 64 --lr 1e-3 --clip_grad 2
python train_gnn.py --data $PROC --out $OUT  --lr 5e-3 --clip_grad 2


#python train_gsnn.py --data ../output/exp1/proc/ --out ../output/exp1/ --dropout 0.2 --channels 4 --lr 5e-3 --clip_grad 2
# python train_nn.py --data ../output/exp1/proc/ --out ../output/exp1/ --dropout 0.2 --channels 64 --lr 1e-4 --clip_grad 2 --ignore_cuda
