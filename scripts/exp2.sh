#!/bin/zsh 

# example use: 
###     $ ./exp1.sh 1
### NOTE: the $1 value defines the name suffix, for use calling multiple exp (diff train/test/val splits) 

########## PARAMS #########
NAME=exp2
PATHWAY=R-HSA-194138
DATA=../../data/
OUT=../output/$NAME-$1/
PROC=$OUT/proc/
EPOCHS=100

MAKE_DATA_TIME=04:00:00
MAKE_DATA_CPUS=4
MAKE_DATA_MEM=32G

GSNN_TIME=03:00:00
GSNN_MEM=12G
GSNN_BATCH=50
GSNN_GRES=gpu:1

NN_TIME=02:00:00
NN_MEM=12G
NN_BATCH=256

GNN_TIME=03:00:00
GNN_MEM=12G
GNN_GRES=gpu:1
GNN_BATCH=50
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
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space landmark --targetome_targets >> $PROC/make_data.out

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
