
# [CoRL 2024] ThinkGrasp: A Vision-Language System for Strategic Part Grasping in Clutter

Welcome to the official repository for **ThinkGrasp: A Vision-Language System for Strategic Part Grasping in Clutter**.  

[![arXiv](https://img.shields.io/badge/arXiv-%23B31B1B.svg?style=for-the-badge&logo=arXiv&logoColor=white)](https://arxiv.org/abs/2407.11298)  [![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=o5QHFhI95Qo) [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?style=for-the-badge&logo=GitHub&logoColor=white)](https://github.com/H-Freax/ThinkGrasp)  
![image](https://github.com/user-attachments/assets/373caf62-99eb-44f6-a7e6-65d18e05e37e)
---

## To-Do List
- [x] Simulation Code Cleanup (without VLP)
- [x] Real-World Code Cleanup (without VLP)
- [ ] Write a Complete README  
- [ ] Add Additional Documentation  

---

## Setup  

### Installation Requirements

- **Operating System**: Ubuntu 23.04  
- **Dependencies**:
  - PyTorch: 1.13.1  
  - Torchvision: 0.14.1  
  - CUDA: 11.8  
  - Pybullet (simulation environment)  
- **Hardware**: GTX 3090 x 2 (for the complete version)
  - **Minimum Requirements**:  
    - **Simulation**: NVIDIA GTX 3090 (single GPU) with ~13GB GPU memory.  
    - **Real-World Execution**: NVIDIA GTX 3090 with ~9.38GB GPU memory (LangSAM).  
  - **Recommended Setup**:  
    - Two NVIDIA GTX 3090 GPUs for best performance when running **VLPart**.

### Installation Steps

1. **Create and Activate the Conda Environment**:  
   ```bash
   conda create -n thinkgrasp python=3.8
   conda activate thinkgrasp
   ```

2. **Install PyTorch and Torchvision**:  
   ```bash
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

3. **Allow Deprecated Scikit-learn**:  
   ```bash
   export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
   ```

4. **Install Additional Requirements**:  
   ```bash
   pip install -r requirements.txt
   pip install -r langsam.txt
   ```

5. **Develop Mode Installation**:  
   ```bash
   python setup.py develop
   ```

6. **Install PointNet2**:  
   ```bash
   cd models/graspnet/pointnet2
   python setup.py install
   cd ../knn
   python setup.py install
   cd ../../..
   ```

7. **Install CUDA 11.8**:  
   Download the CUDA installer and run:  
   ```bash
   sudo bash cuda_11.8.0_520.61.05_linux.run
   ```  
   Add the following lines to your `~/.bashrc` file:  
   ```bash
   export CUDA_HOME=/usr/local/cuda-11.8
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```  
   Refresh the shell:  
   ```bash
   source ~/.bashrc
   ```

---

### Assets

Download the processed object models from:  
- [Google Drive Link](https://drive.google.com/drive/folders/10Kyzzhgcnn1WUlQAhUDk9EBmCzk4p-Ar?usp=sharing)  
- [HuggingFace Dataset](https://huggingface.co/datasets/FreaxRuby/Thinkgrasp/tree/main)  

Place the downloaded files in the `assets` folder. Ensure the structure is as follows:  
```
ThinkGrasp
└── assets
    ├── simplified_objects
    ├── unseen_objects_40
    └── unseen_objects
```

---

### Running the Simulation

1. **Log in to WandB**:  
   ```bash
   wandb login
   ```  

2. **Set Your OpenAI API Key**:  
   ```bash
   export OPENAI_API_KEY="sk-xxxxx"
   ```

3. **Start the Simulation**:  
   ```bash
   pip install protobuf==3.20.1
   python simulation_main.py
   ```  

4. **Change Testing Data**:  
   Update the dataset directory in `simulation_main.py` by modifying line 238:  
   ```python
   parser.add_argument('--testing_case_dir', action='store', type=str, default='heavy_unseen/')
   ```
### Running the Realworld Code

```bash
   pip install flask
   python realarm.py
   ```  


---

### Potential Issues of Installation

#### 1. `AttributeError: module 'numpy' has no attribute 'float'`  

- **Cause**: Deprecated usage of `numpy.float`.  
- **Solution**:  
  Update the problematic lines in the file (e.g., `transforms3d/quaternions.py`):  
  ```python
  _MAX_FLOAT = np.maximum_sctype(np.float64)
  _FLOAT_EPS = np.finfo(np.float64).eps
  ```  
  
#### 2. `graspnetAPI` Installation Issue  

**Error**:  
```plaintext
× python setup.py egg_info did not run successfully.
│ exit code: 1
╰─> [18 lines of output]
The 'sklearn' PyPI package is deprecated, use 'scikit-learn' rather than 'sklearn' for pip commands.
```

**Solution**:  
Allow deprecated scikit-learn compatibility by exporting the following environment variable:  
```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

---

#### 3. CUDA Compatibility Issue  

**Error**:  
```plaintext
RuntimeError: CUDA error: no kernel image is available for execution on the device.
```

**Solution**:  
Ensure the installed PyTorch version matches your CUDA version. For CUDA 11.8, use:  
```bash
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

---

#### 4. Additional Dependencies  

If you still encounter errors, install the following dependencies:  
1. Install Python development tools:  
   ```bash
   sudo apt-get install python3-dev
   ```

2. Install GCC and G++ compilers via Conda:  
   ```bash
   conda install gxx_linux-64
   conda install gcc_linux-64
   ```

3. Install Ray and GroundingDINO:  
   ```bash
   pip install ray
   pip install https://github.com/IDEA-Research/GroundingDINO/archive/refs/tags/v0.1.0-alpha2.tar.gz
   ```

4. Clone and install GroundingDINO:  
   ```bash
   cd langsam
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   pip install -e .
   ```

---

#### 5. CUDA Installation  

Install CUDA 11.8 using the downloaded installer:  
```bash
sudo bash cuda_11.8.0_520.61.05_linux.run
```

Add the following lines to your `~/.bashrc` file:  
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```  

Refresh the shell:  
```bash
source ~/.bashrc
```

---

#### 6. Vision-Language Processing (VLP) Setup  

If you plan to use Vision-Language Processing (VLP):  

1. Install additional requirements:  
   ```bash
   pip install -r vlp_requirements.txt
   ```

2. Download the required `.pth` files:  
   ```bash
   cd VLP
   wget https://github.com/Cheems-Seminar/grounded-segment-any-parts/releases/download/v1.0/swinbase_part_0a0000.pth
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

3. Place the downloaded files in the appropriate directory (`som/downloaddata`).



## Comparison with Vision-Language Grasping (VLG)  

If you want to compare with **VLG**, download the repository from [VLG GitHub](https://github.com/xukechun/Vision-Language-Grasping) and replace the test data and assets.

---

## Citation

If you find this work useful, please consider citing:  
```bibtex
@misc{qian2024thinkgrasp,
  title={ThinkGrasp: A Vision-Language System for Strategic Part Grasping in Clutter},
  author={Yaoyao Qian and Xupeng Zhu and Ondrej Biza and Shuo Jiang and Linfeng Zhao and Haojie Huang and Yu Qi and Robert Platt},
  year={2024},
  eprint={2407.11298},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
