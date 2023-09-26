'''
This script is intended to generate plots comparing the `local` advantage (e.g., individual gene performances) between two models. 

example usage: 
	$ python make_gene_perf_comp_plots.py --uid1 ../output/exp3-1/<UID1> --uid2 ../output/exp3-1/<UID2> --out ../output/
'''

import argparse
import torch 
import numpy as np 
from torch.utils.data import DataLoader 
import sys 
sys.path.append('../.')
from src.models.GSNN import GSNN
from src.data.LincsDataset import LincsDataset
from src.models import utils 

def get_args(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--path1", type=str,
                        help="model 1")

    parser.add_argument("--path2", type=str,
                        help="model 2")

    parser.add_argument("--out", type=str,
                        help="output dir")
    
    parser.add_argument("--data", type=str, help='proc data dir') 

    return parser.parse_args() 

if __name__ == '__main__': 

    args = get_args()

    # load model 1 
    model1 = torch.load(args.path1) 

    # load model 2 
    model2 = torch.load(args.path2)

    # load val loader 
    val_ids = np.load(f'{args.data}/val_obs.npy', allow_pickle=True)
    val_dataset = LincsDataset(root=f'{args.data}', sig_ids=val_ids)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=False)

    # predict
    if torch.cuda.is_available() and not args.ignore_cuda: 
        device = 'cuda'
    else: 
        device = 'cpu' 

   # for metric in metrics : 
       # compute metrics 

       # plot m1 vs m2 

       # save plot (uid1-uid2-metric)  

