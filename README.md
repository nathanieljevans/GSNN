# Graph Structured Neural Networks (GSNN)

## Overview 

The GSNN method is a way of including prior knowledge of latent variable interactions directly into neural architecture.

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

The figures and analysis presented in the preprint can be run using the code available from this [release](https://github.com/nathanieljevans/GSNN/releases/tag/v1.0.0). We have since migrated much of the analysis for the GSNN paper to this auxillary [library](https://github.com/nathanieljevans/gsnn-lib). This library is intended for users who would like to apply the GSNN method to their own data. 

## Getting Started

Create the `conda/mamba` python environment and install the GSNN package: 
```bash 
$ mamba env create -f environment.yml 
$ conda activate gsnn 
(gsnn) $ pip install -e .
```

## Release of version 0.2 

- Improved `SparseLinear` graph-batching (faster)
- Implemented (optional) gradient checkpointing which markedly reduces memory requirements (as much as ~num_layers X memory reduction; ~40% increase runtime). 
- Migrated the perturbation biology analysis code to auxillary repo; "gsnn-lib" 
- added `\examples\` intended to help users understand the GSNN behavior and use-cases
- added GSNN option for `batch` normalization 


