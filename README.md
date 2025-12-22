# BAPA-Net: Background-Aligned Prototype Aggregation

### Official PyTorch Implementation of "Cross-Domain Few-Shot Object Detection in Remote Sensing via Background-Aligned Prototype Aggregation"

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

BAPA-Net is a synergistic architecture designed to fortify cross-domain representational robustness in remote sensing. It addresses the challenges of geospatial domain shifts and intricate background interference by integrating three pivotal mechanisms: **CBFA** for background alignment, **SMPA** for prototype refinement, and **PRKA** for knowledge aggregation.

---

## 📢 Project Status
- ✅ **Core code** has been publicly released.
- 📌 The **complete source code** will be open-sourced upon acceptance of the paper.

---

## 💻 Environment Setup
We recommend using **Anaconda** or **Miniconda** to manage the software environment for reproducibility.

### Installation Steps
1. **Create and Activate Conda Environment**:
   ```bash
   # Initialize a clean Python 3.9 environment
   conda create -n cdfsod python=3.9 -y
   conda activate cdfsod
2. **Install Project Dependencies**:
   ```bash
    # Install core dependencies and the package in editable mode
    pip install -r requirements.txt
    pip install -e ./


## 📂 Datasets & Weights
BAPA-Net uses MS-COCO as the source domain and evaluates on six diverse remote sensing datasets: DIOR, NWPU VHR-10, UAV, MLCD, UODD, and SSDD.

**🔗 Data Download**
Baidu Netdisk: **[https://pan.baidu.com/s/1OmH8d9vxaTk1pVEXMk4thg?pwd=4ru6 Extraction Code: 4ru6)**


## 🏗️ Directory Organization
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

## 🚀 Preparatory Work & Execution

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
  
## 🤝 Acknowledgements
This research is built upon several foundational open-source projects. We extend our gratitude to the authors of:

[DE-ViT](https://github.com/mlzxy/devit) and [CD-ViTO](https://github.com/lovelyqian/CDFSOD-benchmark): For the core few-shot framework and evaluation protocols.

[ViTDeT](https://github.com/ViTAE-Transformer/ViTDet), [Detic](https://github.com/facebookresearch/Detic): For the benchmarking codebases.

© 2024 BAPA-Net Team. Distributed under the Apache 2.0 License.
