#!/bin/sh 

#
# MAPK family signaling cascades (R-HSA-5683057)
# LINCS SPACE: landmark 
# Drug Targets: CLUE + Targetome
# extended GRN: No 
#
#
# input nodes: 3226
# output nodes: 462
# function nodes: 1349
# obs: 29774
#

PATHWAY=R-HSA-5683057
DATA=../../data/
OUT=../output/exp1/
PROC=$OUT/proc/
EPOCHS=100

rm -r $OUT
mkdir $OUT 
mkdir $PROC

conda activate gsnn 
python make_data.py --data $DATA --out $PROC --pathways R-HSA-9006934 --feature_space landmark --targetome_targets 

echo 'submitting gsnn jobs...'
sbatch batched_gsnn.sh $PROC $OUT $EPOCHS

echo 'submitting nn jobs...'
sbatch batched_nn.sh $PROC $OUT $EPOCHS

echo 'submitting gnn jobs...'
sbatch batched_gnn.sh $PROC $OUT $EPOCHS



#python train_gsnn.py --data $PROC --out $OUT --dropout $DO --channels 4 --layers 10 --lr $LR --clip_grad $CG --epochs $EPOCHS
#python train_gsnn.py --data $PROC --out $OUT --dropout $DO --channels 4 --layers 10 --lr $LR --clip_grad $CG --randomize --epochs $EPOCHS
#python train_nn.py --data $PROC --out $OUT --dropout $DO --channels 64 --lr $LR --clip_grad $CG --epochs $EPOCHS
#python train_nn.py --data $PROC --out $OUT --dropout $DO --channels 64 --lr $LR --clip_grad $CG --cell_agnostic --epochs $EPOCHS
#python train_gnn.py --data $PROC --out $OUT --dropout $DO --lr $LR --clip_grad $CG --epochs $EPOCHS
#python train_gnn.py --data $PROC --out $OUT --dropout $DO --lr $LR --clip_grad $CG --epochs $EPOCHS --randomize