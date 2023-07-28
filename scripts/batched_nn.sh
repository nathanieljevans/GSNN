#!/bin/bash

# example use: 
# sbatch batched_nn.sh $PROC $OUT $EPOCHS

jobid=0
for lr in "0.1" "0.01", "0.001"; do
    for do in "0.0" "0.25" "0.5"; do 
        for c in 32 64 128; do
                jobid=$jobid+1

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=evans_$jobid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err

conda activate gsnn 
cd /home/exacloud/gscratch/NGSdev/evans/GSNN/scripts/
python train_nn.py --data $1 --out $2 --dropout $do --channels $c --lr $lr --epochs $3
python train_nn.py --data $1 --out $2 --dropout $do --channels $c --lr $lr --epochs $3 --cell_agnostic

EOF
done