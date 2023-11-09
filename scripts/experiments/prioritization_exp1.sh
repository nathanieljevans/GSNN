#!/bin/sh 

PATHWAY=PATHWAY="R-HSA-177929 R-HSA-1489509 R-HSA-1257604 R-HSA-5673001 R-HSA-1227986 R-HSA-109606 R-HSA-6806003 R-HSA-202131 R-HSA-6807070 R-HSA-6807070 R-HSA-5673001"
PROC=../output/exp1/proc/
DATA=../../data/
FOLD_DIR=../output/exp1/FOLD-1/
GSNN_OUT=../output/exp1/FOLD-1/GSNN/
NN_OUT=../output/exp1/FOLD-1/NN/

cd ..

# rm -r $PROC 
# mkdir $PROC
# remove GSNN?NN_out
#python make_data.py --data $DATA --out $PROC --pathways $PATHWAY --feature_space landmark --targetome
#python create_data_splits.py --data $DATA --proc $PROC --out $FOLD_DIR

#python train_gsnn.py --data $PROC --fold $FOLD_DIR --out $GSNN_OUT --dropout 0 --channels 10 --lr 1e-2 --epochs 100 --batch 100 --layers 15 --add_function_self_edges
#GSNN_UID=$(ls -d $GSNN_OUT/*/ | head -n 1)
#echo 'GSNN UID:' $GSNN_UID
#python train_viabnn.py --data $DATA --proc $PROC --fold $FOLD_DIR --model_dir $GSNN_UID --model_name model-100.pt --epochs 400 --channels 1000 --target_distribution beta --N 3 --target cell_viability --use_ensemble_mixture --lr 1e-3 --batch 1024
#python prioritize.py --proc $PROC --uid_dir $GSNN_UID --goals_path ../output/disease_prioritization_goals.csv --model model-100.pt --verbose

python train_nn.py --data $PROC --fold $FOLD_DIR --out $NN_OUT --dropout 0 --channels 1000 --lr 1e-3 --epochs 100 --batch 512
NN_UID=$(ls -d $NN_OUT/*/ | head -n 1)
echo 'NN UID: ' $NN_UID
python train_viabnn.py --data $DATA --proc $PROC --fold $FOLD_DIR --model_dir $NN_UID --model_name model-100.pt --epochs 400 --channels 1000 --target_distribution beta --N 3 --target cell_viability --use_ensemble_mixture --lr 1e-3 --batch 1024
python prioritize.py --proc $PROC --uid_dir $NN_UID --goals_path ../output/disease_prioritization_goals.csv --model model-100.pt --verbose

##python prioritize.py --proc ../output/exp1/proc/ --uid_dir ../output/exp1/FOLD-1/GSNN//8c62d604-d99c-48d5-9bf0-9a8e9d30c644/ --goals_path ../output/breast_subtype_goals.csv --model model-100.pt --verbose


