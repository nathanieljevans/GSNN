#!/bin/zsh
# example use:
# ./batched_gnn.sh $PROC $OUT $EPOCHS $TIME $MEM $BATCH 
# ./batched_gnn.sh ../output/exp1-1/proc/ ../output/gnn_test/ 10 00:30:00 12G 50 
  

PROC=$1 
OUT=$2
EPOCHS=$3
TIME=$4
MEM=$5
BATCH=$6
GPU=$7
FOLD_DIR=$8

echo "PROC" $PROC 
echo "OUT" $OUT 
echo "EPOCHS" $EPOCHS
echo "TIME" $TIME 
echo "MEM" $MEM
echo "BATCH" $BATCH
echo "GPU" $GPU

mkdir $OUT

# make slurm log dir
OUT2=$OUT/slurm_logs__GNN/
if [ -d "$OUT2" ]; then
	echo "slurm output log dir exists. Erasing contents..."
        rm -r "$OUT2"/*
else
	echo "slurm output log dir does not exist. Creating..."
        mkdir "$OUT2"
fi

jobid=0
for lr in 0.001; do
    for do in 0; do
        for c in 128; do
            for conv in GIN; do
                for layers in 3; do


jobid=$((jobid+1))

echo "submitting job: nn (lr=$lr, do=$do, c=$c)"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gnn$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=$GPU
#SBATCH --time=$TIME
#SBATCH --mem=$MEM
#SBATCH --partition=gpu
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

source ~/.zshrc
conda activate gsnn 
cd /home/exacloud/gscratch/NGSdev/evans/GSNN/scripts/
python train_gnn.py --fold $FOLD_DIR --data $PROC --out $OUT --layers $layers --dropout $do --channels $c --lr $lr --epochs $EPOCHS --gnn $conv --batch $BATCH

EOF
done
done
done
done
done 
