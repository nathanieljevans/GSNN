#!/bin/zsh 

# example use: 
###     $ ./exp1.sh 1
### NOTE: the $1 value defines the name suffix, for use calling multiple exp (diff train/test/val splits) 

# Signaling by EGFR (R-HSA-177929) 
#     linked downstream pathways (that aren't included in `signaling by EGFR` geneset): 
#     	- DAG and IP3 signaling - R-HSA-1489509
#     	- PIP3 activates AKT signaling - R-HSA-1257604
#     		- Intrinsic Pathway for Apoptosis - R-HSA-109606
#    		- Regulation of TP53 Expression and Degradation - R-HSA-6806003
#     		- Metabolism of nitric oxide: NOS3 activation and regulation - R-HSA-202131
#     		- PTEN Regulation - R-HSA-6807070 
#     		- MTOR signaling - R-HSA-165159
#     	- RAF/MAP kinase cascade - R-HSA-5673001 
#
# Signaling by ERBB2 - R-HSA-1227986
#       - Signaling by EGFR 
#       
# This experiment tests if including these downstream "linked" pathways improves the model performance relative to NN.
# same as 12, but adding second level of dependants + ERBB2 
########## PARAMS #########

NAME=exp1
PATHWAY="R-HSA-177929 R-HSA-1489509 R-HSA-1257604 R-HSA-5673001 R-HSA-1227986 R-HSA-109606 R-HSA-6806003 R-HSA-202131 R-HSA-6807070 R-HSA-6807070 R-HSA-5673001"
DATA=../../data/
OUT=../output/$NAME-$1/
PROC=$OUT/proc/
EPOCHS=100

MAKE_DATA_TIME=06:00:00
MAKE_DATA_CPUS=8
MAKE_DATA_MEM=32G
FEATURE_SPACE="landmark"
TARGETOME="--targetome_targets"
STITCH=""
FULL_GRN=""

GSNN_TIME=1-12:00:00
GSNN_MEM=32G
GSNN_BATCH=50
GSNN_GRES=gpu:1

NN_TIME=08:00:00
NN_MEM=16G
NN_BATCH=256

GNN_TIME=16:00:00
GNN_MEM=32G
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

cd .. 
pwd
echo 'making data...' 
source ~/.zshrc
conda activate gsnn 

echo 'removing out dir and making proc dir...'
rm -r $OUT
mkdir $OUT
mkdir $PROC

#                                                                                               # Flags->
python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space $FEATURE_SPACE $TARGETOME $STITCH $FULL_GRN >> $PROC/make_data.out

if [ -e "$PROC/make_data_completed_successfully.flag" ]; then
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
	#                                        HH:MM:SS MEM   BTCH
	./batched_gnn.sh $PROC $OUT/GNN/ $EPOCHS $GNN_TIME $GNN_MEM $GNN_BATCH
else 
	echo "make_data.py did not complete successfully. no model batch scripts submitted."
fi
EOF

