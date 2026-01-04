# CAPPM: Conflict-Aware Transformer based Diagnosis of Alzheimer's Disease by integrating Plasma Proteomics and MRI Data

For neurodegenerative disorders, plasma proteomic data and neuroimages have reflected disease dynamics at microscopic and macroscopic cales, respectively. Integrating consistency and specificity between heterogeneous datasets is able to improves prediction performance and reliability by integrating information from various scales. Existing multi-modal learning frameworks have encountered problems in modal conflict and inconsistency. In this work, we propose a conflict-aware Transformer (CAPPM) in AD diagnosis by integrating plasma proteomics and MRI data. This CAPPM method introduces a dual-branch progressive convergent network (DPCN) to extract uni-modal features and fuse heterogeneous representations into a shared latent space. For proteomics and MRI data, the gradient-based conflict-aware (GCA) module was proposed. Negative cosine similarity between gradient angles was defined as gradient conflict, which is regarded as a special case of modal coflict. Additionally, the modal conflict-coordination (MCC) module has been employed to alleviate gradient conflict, enhancing the expressiveness of joint latent representations. In validation experiments about multi-omics datasets, the proposed CAPPM method has demonstrated superior performance over existing multi-modal learning methods, demonstrating that a reasonable level of gradient conflict is useful to boost model performance and generalization. The CAPPM method has provided an effective and promising multi-modal method for the diagnosis of Alzheimer's disease.

## Architecture

The framework of the proposed CAPPM method. (A) The role of DPCN is to capture cross-scale interactions between proteomics and MRI data. The DPCN Module consists of D stacked Shared-MRI-Proteomics Fusion (SMPF) layers. (B) The GCA module is designed to perceive the cross-scale conflict scenarios. (C) At each SMPF layer of the DPCN module, the shared latent representation undergoes sequential fusion with individual modalities. 

![Fig1_v6_01](figure/Fig1_v6_01.pdf)



Overview of the MCC block. Left: Modal conflicts can be characterized by the gradients of uni-modal loss and multi-modal loss.
Right: In the gradient space, $g^m$ and $g^u$ represent the gradients of uni-modal loss and multi-modal loss, respectively, while $g^{MCC}$ denotes the coordinated gradient. 

![Fig2_v6_01](figure/Fig2_v6_01.pdf)

## Install

We train the SIMN model under the Ubuntu22.04, and graphics card is RTx4090.

```
conda create -n SIMN python==3.8
conda activate SIMN
```

## Requirements

```
pip install -r requirements.txt
```

## Data availability

Multi-omics datasets used in this study were available from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database {https://ida.loni.usc.edu/} and a comprehensive and accessible Alzheimer’s disease patient-level dataset (ANMerge) {https://www.synapse.org/Synapse:syn22252881}. As such, researchers within ADNI and ANMerge contributed and/or provided data for the design and implementation of ADNI and ANMerge, but were not involved in the analysis of this work.

## Usage

1、Run the processing/{datasets}/overlap.ipynb file to generate a sample-aligned data set and

split the training set, validation set, and test set.

2、Run the processing/{datasets}/process_plasma.ipynb file to make full use of the data set that

cannot be sample aligned as a single-modal training set.

2、Run the processing/{datasets}/split_{datasets}_2D_3class.ipynb file to make full use of the data

set that cannot be sample aligned as a single-modal training set.

The obtained files are located under processed_data/datasets/

## Training

Run the cappm/{datasets}.ipynb file to get the Multi-modal model.


The obtained files are located under cappm/{datasets}/

