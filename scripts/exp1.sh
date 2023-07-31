#!/bin/zsh 


#SBATCH --job-name=exp1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=EXP1.%j.out
#SBATCH --error=EXP1.%j.err


#
# EGFR Signaling 
# LINCS SPACE: landmark 
# Drug Targets: CLUE + Targetome
# extended GRN: No 
#

########## PARAMS #########
PATHWAY=R-HSA-177929
DATA=../../data/
OUT=../output/exp1-1/
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
./batched_gsnn.sh $PROC $OUT/GSNN/ $EPOCHS

echo 'submitting nn jobs...'
mkdir $OUT/NN/
./batched_nn.sh $PROC $OUT/NN/ $EPOCHS

echo 'submitting gnn jobs...'
mkdir $OUT/GNN/
./batched_gnn.sh $PROC $OUT/GNN/ $EPOCHS


