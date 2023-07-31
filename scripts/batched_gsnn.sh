#!/bin/zsh
# example use:
# sbatch batched_nn.sh $PROC $OUT $EPOCHS

# make slurm log dir 
OUT2=$2/slurm_logs__GSNN/
if [ -d "$OUT2" ]; then
	echo "slurm output log dir exists. Erasing contents..."
        rm -r "$OUT2"/*
else
	echo "slurm output log dir does not exist. Creating..."
        mkdir "$OUT2"
fi

jobid=0
for lr in 0.01 0.001; do
    for do in 0.0 0.25 0.5; do 
        for c in 2 5 10; do
		for b in 50 100; do

jobid=$((jobid+1))

echo "submitting job: nn (lr=$lr, do=$do, c=$c, b=$b)"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gsnn$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --output=$OUT2/log.%j.out
#SBATCH --error=$OUT2/log.%j.err

source ~/.zshrc
conda activate gsnn 
cd /home/exacloud/gscratch/NGSdev/evans/GSNN/scripts/
python train_gsnn.py --data $1 --out $2 --dropout $do --channels $c --lr $lr --epochs $3 --batch $b
python train_gsnn.py --data $1 --out $2 --dropout $do --channels $c --lr $lr --epochs $3 --batch $b --randomize

EOF
done
done
done
done
