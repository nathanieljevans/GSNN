#!/bin/zsh 
# example use: 
###     $ sbatch exp1.sh

#SBATCH --job-name=exp4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=./SLURM_OUT/EXP4.%j.out
#SBATCH --error=./SLURM_OUT/EXP4.%j.err


#
# MAPK family signaling cascades
# LINCS SPACE: landmark + best inferred 
# Drug Targets: CLUE + Targetome + STITCH 
# extended GRN: No 
#

########## PARAMS #########
PATHWAY=R-HSA-5683057
DATA=../../data/
OUT=../output/exp4-1/
PROC=$OUT/proc/
EPOCHS=100
##########################

echo 'removing out dir and making proc dir...'
rm -r $OUT
mkdir $OUT 
mkdir $PROC

echo 'making data...' 
source ~/.zshrc
conda activate gsnn 
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space landmark best-inferred --targetome_targets --stitch_targets >> $PROC/make_data.out

echo 'submitting gsnn jobs...'
mkdir $OUT/GSNN/
#                                          D-HH:MM:SS MEM BTCH GRES        
./batched_gsnn.sh $PROC $OUT/GSNN/ $EPOCHS 1-12:00:00 16G 100 gpu:1 

echo 'submitting nn jobs...'
mkdir $OUT/NN/
#                                      HH:MM:SS MEM BTCH
./batched_nn.sh $PROC $OUT/NN/ $EPOCHS 04:00:00 12G 256

echo 'submitting gnn jobs...'
mkdir $OUT/GNN/
#                                        HH:MM:SS MEM GRES  BTCH
./batched_gnn.sh $PROC $OUT/GNN/ $EPOCHS 12:00:00 12G gpu:1 50


