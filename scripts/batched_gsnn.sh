#!/bin/zsh
# example use:
# ./batched_gsnn.sh $PROC $OUT $EPOCHS $TIME $MEM $BATCH $GPU
# ./batched_gsnn.sh ../output/exp1-1/proc/ ../output/gpu_test/ 10 00:10:00 8G 50 gpu:rtx2080:1 

PROC=$1   # path to processed data directory 
OUT=$2    # path to output directory 
EPOCHS=$3 # number of training epochs to run 
TIME=$4   # amount of time to request for the slurm job (hours); e.g., 01:00:00 -> 1 hour 
MEM=$5    # amount of memory to request for the slurm job (GB); should be in format xG, e.g., 16G -> 16 GB 
BATCH=$6  # batch size to use, smaller batches will use less memory, but may affect optimization results. 
GPU=$7    # gpu to request for the slurm job; should be in format: #SBATCH "gpu:rtx2080:1" -> 1 rtx2080 GPU 

mkdir $OUT

echo "PROC=$PROC"
echo "OUT=$OUT" 
echo "EPOCHS=$EPOCHS" 
echo "TIME=$TIME"
echo "MEM=$MEM" 
echo "BATCH=$BATCH" 
echo "GPU=$GPU"

# make slurm log dir 
OUT2=$OUT/slurm_logs__GSNN/
if [ -d "$OUT2" ]; then
	echo "slurm output log dir exists. Erasing contents..."
        rm -r "$OUT2"/*
else
	echo "slurm output log dir does not exist. Creating..."
        mkdir "$OUT2"
fi

jobid=0
# HYPER-PARAMETER GRID SEARCH 
for lr in 0.05 0.005; do
    for do in 0 0.25 0.5; do 
        for c in 5 10; do
	    for lay in 10 15; do

jobid=$((jobid+1))

echo "submitting job: GSNN  (lr=$lr, do=$do, c=$c)"

# SUBMIT SBATCH JOB 

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gsnn$jobid
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
python train_gsnn.py --data $PROC --out $OUT --dropout $do --channels $c --lr $lr --epochs $EPOCHS --batch $BATCH --layers $lay
python train_gsnn.py --data $PROC --out $OUT --dropout $do --channels $c --lr $lr --epochs $EPOCHS --batch $BATCH --layers $lay --randomize

EOF
done
done
done
done
