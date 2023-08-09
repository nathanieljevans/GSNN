#!/bin/zsh 
# example use: 
###     $ sbatch exp2.sh

#SBATCH --job-name=exp2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=./SLURM_OUT/EXP2.%j.out
#SBATCH --error=./SLURM_OUT/EXP2.%j.err


#
# Signaling by Receptor Tyrosine Kinases
# LINCS SPACE: landmark 
# Drug Targets: CLUE + Targetome
# extended GRN: No 
#

########## PARAMS #########
PATHWAY=R-HSA-9006934
DATA=../../data/
OUT=../output/exp2-1/
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
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space landmark --targetome_targets >> $PROC/make_data.out

echo 'submitting gsnn jobs...'
mkdir $OUT/GSNN/
#                                          D-HH:MM:SS MEM BTCH GRES        
./batched_gsnn.sh $PROC $OUT/GSNN/ $EPOCHS 2-00:00:00 16G 100 gpu:1 

echo 'submitting nn jobs...'
mkdir $OUT/NN/
#                                      HH:MM:SS MEM BTCH
./batched_nn.sh $PROC $OUT/NN/ $EPOCHS 02:00:00 12G 256

echo 'submitting gnn jobs...'
mkdir $OUT/GNN/
#                                        HH:MM:SS MEM GRES  BTCH
./batched_gnn.sh $PROC $OUT/GNN/ $EPOCHS 12:00:00 12G gpu:1 50


