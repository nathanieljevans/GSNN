#!/bin/sh 

# Summary: Medium size pathway, testing STITCH targets
#
# MAPK family signaling cascades (R-HSA-5683057)
# LINCS SPACE: landmark 
# Drug Targets: CLUE + Targetome + STITCH
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
OUT=../output/exp2/
PROC=$OUT/proc/
DO=0.33
CG=3
LR=1e-2
EPOCHS=100

rm -r $OUT
mkdir $OUT 
mkdir $PROC

python make_data.py --data $DATA --out $PROC --pathways R-HSA-9006934 --feature_space landmark --targetome_targets --stitch_targets

python train_gsnn.py --data $PROC --out $OUT --dropout $DO --channels 4 --layers 10 --lr $LR --clip_grad $CG --epochs $EPOCHS
python train_gsnn.py --data $PROC --out $OUT --dropout $DO --channels 4 --layers 10 --lr $LR --clip_grad $CG --randomize --epochs $EPOCHS
python train_nn.py --data $PROC --out $OUT --dropout $DO --channels 64 --lr $LR --clip_grad $CG --epochs $EPOCHS
python train_nn.py --data $PROC --out $OUT --dropout $DO --channels 64 --lr $LR --clip_grad $CG --cell_agnostic --epochs $EPOCHS
python train_gnn.py --data $PROC --out $OUT --dropout $DO --lr $LR --clip_grad $CG --epochs $EPOCHS
python train_gnn.py --data $PROC --out $OUT --dropout $DO --lr $LR --clip_grad $CG --epochs $EPOCHS --randomize