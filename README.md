# Graph Structured Neural Networks (GSNN)

## Overview 

The GSNN method is a algorithm that integrates prior knowledge of latent variable interactions directly into neural architecture.

```
@article {Evans2024.02.28.582164,
	author = {Nathaniel J. Evans and Gordon B. Mills and Guanming Wu and Xubo Song and Shannon McWeeney},
	title = {Graph Structured Neural Networks for Perturbation Biology},
	elocation-id = {2024.02.28.582164},
	year = {2024},
	doi = {10.1101/2024.02.28.582164},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/02/29/2024.02.28.582164},
	eprint = {https://www.biorxiv.org/content/early/2024/02/29/2024.02.28.582164.full.pdf},
	journal = {bioRxiv}
}
```

The figures and analysis presented in the preprint can be run using the code available from this [release](https://github.com/nathanieljevans/GSNN/releases/tag/v1.0.0). We have since migrated much of the analysis for the GSNN paper to this auxillary [library](https://github.com/nathanieljevans/gsnn-lib).

## Getting Started

Create the `conda/mamba` python environment and install the GSNN package: 
```bash 
$ mamba env create -f environment.yml 
$ conda activate gsnn 
(gsnn) $ pip install -e .
```

