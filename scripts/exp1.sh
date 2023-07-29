#!/bin/sh 

#
# EGFR Signaling 
# LINCS SPACE: landmark 
# Drug Targets: CLUE + Targetome
# extended GRN: No 
#


# python make_data.py --data ../../data/ --out ../output/exp1/proc/ --pathways R-HSA-177929 --feature_space landmark --targetome_targets 
# python train_gsnn.py --data ../output/exp1/proc/ --out ../output/exp1/ --dropout 0.33 --channels 4 --layers 10 --lr 1e-1 --epochs 100
# python train_nn.py --data ../output/exp1/proc/ --out ../output/exp1/ --dropout 0.33 --channels 64 --lr 1e-3 --epochs 100


PATHWAY=R-HSA-177929
DATA=../../data/
OUT=../output/exp1/
PROC=$OUT/proc/
EPOCHS=100

rm -r $OUT
mkdir $OUT 
mkdir $PROC

conda activate gsnn 
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space landmark --targetome_targets 

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