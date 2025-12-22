# BAPA-Net
PyTorch implementation of "[BAPA-Net: Cross-Domain Few-Shot Object Detection in Remote Sensing via Background-Aligned Prototype Aggregation]"

The core code has been published, and the full code will be open source after the paper is accepted. Thank you for your attention!

## The Environments
The evaluation environments we adopted are recorded in the following section. Below are the system requirements and setup instructions for reproducing the evaluation environment.

### Required Environment Setup
We suggest using Anaconda for environment management. Here's how to set up the environment for the challenge:

- **Step 1**: conda environment create:
  ```bash
    conda create -n cdfsod python=3.9
    conda activate cdfsod
- **Step 2**: install other libs:
  ```bash
    cd BAPA-Net
    pip install -r requirements.txt
    pip install -e ./
or take it as a reference based on your original environments.

## The Train and Validation Datasets
We take COCO as source data and DIOR, NWPU VHR-10,UAV,MLCD,UODD and SSDD as validation datasets.

The target datasets could be easily downloaded in the following links:**[https://pan.baidu.com/s/1OmH8d9vxaTk1pVEXMk4thg?pwd=4ru6 Extraction Code: 4ru6)**


After downloading all the necessary validation datasets, make sure they are organized as follows:

```bash
|BAPA-Net/datasets/
|--DIOR/
|   |--annotations
|   |--test
|   |--train
|--UAV/
|   |--annotations
|   |-images
|   |--JPEGImages
|--......
```
And the weights should be organized as follows:
```bash
|BAPA-Net/weights/
|--trained/
|   |--vitl_0089999.pth
|--background/
|   |--background_prototypes.vitl14.pth
|   |--DIOR_1shot
|   |--DIOR_5shot
|   |--DIOR_10shot
```
And the prototypes_init should be organized as follows:
```bash
|BAPA-Net/prototypes_init/
|--DIOR_1shot.vitl14.bbox.p1.sk.pkl
|--DIOR_1shot.vitl14.bbox.pkl
|--DIOR_1shot.vitl14.bbox_ori.pkl
|--DIOR_5shot.vitl14.bbox.p1.sk.pkl
|--DIOR_5shot.vitl14.bbox.pkl
|--DIOR_5shot.vitl14.bbox_ori.pkl
|--DIOR_10shot.vitl14.bbox.p1.sk.pkl
|--DIOR_10shot.vitl14.bbox.pkl
|--DIOR_10shot.vitl14.bbox_ori.pkl
```
## Preparatory Work Before Training
- **Step 1**: Generate cache_models:
  ```bash
    bash bulid_cache.sh
- **Step 2**: Build Prototypes:
  ```bash
    bash bulid_prototypes.sh
- **Step 3**: Build Background Prototypes:
  ```bash
    bash bulid_backpround_prototypes.sh
- **Step 4**: Run BAPA-Net:
  ```bash
    bash main_results.sh  
  

## Acknowledgement
Our work is built upon [DE-ViT](https://github.com/mlzxy/devit) and [CD-ViTO](https://github.com/lovelyqian/CDFSOD-benchmark), and also we use the codes of [ViTDeT](https://github.com/ViTAE-Transformer/ViTDet), [Detic](https://github.com/facebookresearch/Detic) to test them under this new benchmark. Thanks for their work.
