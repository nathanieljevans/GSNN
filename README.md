# Graph Structured Neural Networks (GSNN)

Nathaniel Evans 
evansna@ohsu.edu

## Getting Started

```bash 
$ mamba env create -f environment.yml 
$ conda activate gsnn 
(gsnn) $
```

```bash 
$ ./get_data.sh /path/to/download/dir/
```

```bash 
(gsnn) $ python make_data.py --data /path/to/download/dir/ --out /path/to/processed/dir/ --pathways R-HSA-9006934 --feature_space landmark best-inferred --stitch_targets --targetome_targets
```

```bash 
(gsnn) $ python train_gsnn.py --data --out /path/to/processed/dir/ --dropout 0.2 --channels 4 --lr 5e-3 --clip_grad 2
```

```bash 
(gsnn) $ python train_gnn.py --data --out /path/to/processed/dir/
```

```bash 
(gsnn) $ python train_nn.py --data --out /path/to/processed/dir/
```

NOTE: use ```$ python <fn> --help``` to get optional command line arguments. 


## Reactome Pathways Suggestions

| **Reactome ID** 	| **Reactome Level** | **Name**                               	  | **Size** 	|
|-----------------	|------------------- | ------------------------------------------ |------------ |
| R-HSA-162582    	| 0                  | Signal Transduction                    	  | 2584     	|
| R-HSA-9006934   	| 1                  | Signaling by Receptor Tyrosine Kinases 	  | 519      	|
| R-HSA-5683057   	| 1                  | MAPK family signaling cascades         	  | 327      	|
| R-HSA-372790      | 1                  | Signaling by GPCR                          |             | 
| R-HSA-157118      | 1                  | Signaling by NOTCH                         |             |
| R-HSA-195721      | 1                  | Signaling by WNT                           |             |
| R-HSA-201681      | 2                  | TCF dependent signaling in response to WNT |             |
| 
