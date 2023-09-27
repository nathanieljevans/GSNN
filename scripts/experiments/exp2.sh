#!/bin/zsh 

# example use: 
###     $ ./exp2.sh 1
### NOTE: the $1 value defines the name suffix, for use calling multiple exp (diff train/test/val splits) 

# (NOTE: limit to depth of 3)

# Signaling by 
# Death receptor signaling: R-HSA-73887
#       FasL/ CD95L signaling : R-HSA-75157 
#               Caspase activation via Death Receptors in the presence of ligand: R-HSA-140534 
#       TRAIL signaling: R-HSA-75158
#               Caspase activation via Death Receptors in the presence of ligand: R-HSA-140534 
#       TNF signaling: R-HSA-75893
#               Caspase activation via Death Receptors in the presence of ligand: R-HSA-140534 
#               Regulated Necrosis: R-HSA-5218859 
#                       RIPK1-mediated regulated necrosis: R-HSA-5213460
#                       Pyroptosis: R-HSA-5620971 
#                               Intrinsic Pathway for Apoptosis: R-HSA-109606
#                               Interleukin-1 family signaling: R-HSA-446652
#                               Regulation of TLR by endogenous ligand: R-HSA-5686938
#      p75 NTR receptor-mediated signalling: R-HSA-193704
#               Signaling by NTRK1 (TRKA): R-HSA-187037 
#               RAF/MAP kinase cascade: R-HSA-5673001
#               PIP3 activates AKT signaling: R-HSA-1257604
#               NGF-stimulated transcription: R-HSA-9031628
#               DAG and IP3 signaling: R-HSA-1489509

########## PARAMS #########

NAME=exp2
PATHWAY="R-HSA-73887 R-HSA-75157 R-HSA-140534 R-HSA-75158 R-HSA-75893 R-HSA-5218859 R-HSA-5213460 R-HSA-5620971 R-HSA-109606 R-HSA-446652 R-HSA-5686938 R-HSA-193704 R-HSA-187037 R-HSA-5673001 R-HSA-1257604 R-HSA-9031628 R-HSA-1489509"
DATA=../../data/
OUT=../output/$NAME-$1/
PROC=$OUT/proc/
EPOCHS=100
N_FOLDS=3

MAKE_DATA_TIME=08:00:00
MAKE_DATA_CPUS=8
MAKE_DATA_MEM=32G
FEATURE_SPACE="landmark"
TARGETOME="--targetome_targets"
STITCH=""
FULL_GRN=""

GSNN_TIME=1-12:00:00
GSNN_MEM=32G
GSNN_BATCH=25
GSNN_GRES=gpu:v100:1
# BUG: the `p100` GPUs lead to ECC errors: a40 (n=16), v100 (n=32), rtx2080 (n=32) seem to work normally. 

NN_TIME=08:00:00
NN_MEM=16G
NN_BATCH=256

GNN_TIME=12:00:00
GNN_MEM=16G
GNN_BATCH=25
GNN_GRES=gpu:v100:1
# BUG: the `p100` GPUs lead to incosistent ECC errors: a40 (n=16), v100 (n=32), rtx2080 (n=32) seem to work normally.
###########################


sbatch <<EOF
#!/bin/zsh
#SBATCH --job-name=$NAME
#SBATCH --nodes=1
#SBATCH --cpus-per-task=$MAKE_DATA_CPUS
#SBATCH --time=$MAKE_DATA_TIME
#SBATCH --mem=$MAKE_DATA_MEM
#SBATCH --output=./SLURM_OUT/$NAME-%j.out
#SBATCH --error=./SLURM_OUT/$NAME-%j.err

cd .. 
pwd
echo 'making data...' 
source ~/.zshrc
conda activate gsnn 

echo 'removing out dir and making proc dir...'
rm -r $OUT
mkdir $OUT
mkdir $PROC

# create processed data directory 
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space $FEATURE_SPACE $TARGETOME $STITCH $FULL_GRN >> $PROC/make_data.out

if [ -e "$PROC/make_data_completed_successfully.flag" ]; then

	for (( FOLD=0; FOLD<=$N_FOLDS; FOLD++ )); do

		FOLD_DIR="$OUT/FOLD-\$FOLD"
		echo "FOLD DIR: \$FOLD_DIR"
		mkdir \$FOLD_DIR 

		python create_data_splits.py --data $DATA --proc $PROC --out \$FOLD_DIR

		echo 'submitting gsnn jobs...'
		mkdir \$FOLD_DIR/GSNN/
		#                                          HH:MM:SS MEM BTCH GRES        
		./batched_gsnn.sh $PROC \$FOLD_DIR/GSNN/ $EPOCHS $GSNN_TIME $GSNN_MEM $GSNN_BATCH $GSNN_GRES \$FOLD_DIR 

		echo 'submitting nn jobs...'
		mkdir \$FOLD_DIR/NN/
		#                                      HH:MM:SS MEM BTCH
		./batched_nn.sh $PROC \$FOLD_DIR/NN/ $EPOCHS $NN_TIME $NN_MEM $NN_BATCH \$FOLD_DIR 

		echo 'submitting gnn jobs...'
		mkdir \$FOLD_DIR/GNN/
		#                                        HH:MM:SS MEM   BTCH
		./batched_gnn.sh $PROC \$FOLD_DIR/GNN/ $EPOCHS $GNN_TIME $GNN_MEM $GNN_BATCH $GNN_GRES \$FOLD_DIR 

	done 

else 
	echo "make_data.py did not complete successfully. no model batch scripts submitted."
fi
EOF

