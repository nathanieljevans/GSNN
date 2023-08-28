#!/bin/zsh 

# example use: 
###     $ ./exp1.sh 1
### NOTE: the $1 value defines the name suffix, for use calling multiple exp (diff train/test/val splits) 

# Signaling by EGFR (R-HSA-177929) 
#     linked downstream pathways (that aren't included in `signaling by EGFR` geneset): 
#     	- DAG and IP3 signaling - R-HSA-1489509
#     	- PIP3 activates AKT signaling - R-HSA-1257604
#     	- RAF/MAP kinase cascade - R-HSA-5673001 

# This experiment tests if including these downstream "linked" pathways improves the model performance relative to NN.

########## PARAMS #########
NAME=exp12
PATHWAY="R-HSA-177929 R-HSA-1489509 R-HSA-1257604 R-HSA-5673001"
DATA=../../data/
OUT=../output/$NAME-$1/
PROC=$OUT/proc/
EPOCHS=100

MAKE_DATA_TIME=04:00:00
MAKE_DATA_CPUS=8
MAKE_DATA_MEM=32G
FEATURE_SPACE="landmark"
TARGETOME="--targetome_targets"
STITCH=""
FULL_GRN=""

GSNN_TIME=16:00:00
GSNN_MEM=20G
GSNN_BATCH=25
GSNN_GRES=gpu:1

NN_TIME=04:00:00
NN_MEM=12G
NN_BATCH=256

GNN_TIME=08:00:00
GNN_MEM=20G
GNN_GRES=gpu:1
GNN_BATCH=25
###########################

sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=$NAME-$1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$MAKE_DATA_CPUS
#SBATCH --time=$MAKE_DATA_TIME
#SBATCH --mem=$MAKE_DATA_MEM
#SBATCH --output=./SLURM_OUT/$NAME-$1-%j.out
#SBATCH --error=./SLURM_OUT/$NAME-$1-%j.err

#
# EGFR Signaling 
# LINCS SPACE: landmark 
# Drug Targets: CLUE + Targetome
# extended GRN: No 
#

echo 'removing out dir and making proc dir...'
rm -r $OUT
mkdir $OUT 
mkdir $PROC

echo 'making data...' 
source ~/.zshrc
conda activate gsnn 
#                                                                                               # Flags->
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space $FEATURE_SPACE $TARGETOME $STITCH $FULL_GRN >> $PROC/make_data.out

echo 'submitting gsnn jobs...'
mkdir $OUT/GSNN/
#                                          HH:MM:SS MEM BTCH GRES        
./batched_gsnn.sh $PROC $OUT/GSNN/ $EPOCHS $GSNN_TIME $GSNN_MEM $GSNN_BATCH $GSNN_GRES 

echo 'submitting nn jobs...'
mkdir $OUT/NN/
#                                      HH:MM:SS MEM BTCH
./batched_nn.sh $PROC $OUT/NN/ $EPOCHS $NN_TIME $NN_MEM $NN_BATCH

echo 'submitting gnn jobs...'
mkdir $OUT/GNN/
#                                        HH:MM:SS MEM GRES  BTCH
./batched_gnn.sh $PROC $OUT/GNN/ $EPOCHS $GNN_TIME $GNN_MEM $GNN_GRES $GNN_BATCH

EOF
