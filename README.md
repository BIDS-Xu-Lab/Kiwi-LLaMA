# Information Extraction from Clinical Notes: Are We Ready for Adopting Large Language Models?

An experiment to determine if we can create a system to feasibly extract reliably sufficient information from clinical notes.

## Overview

This repository contains the code and resources for Kiwi, a LLM-based clinical Information Extraction (IE) system designed to extract medical problem, treatment, test, and drug entities, and their modifiers from clinical note.

## Requirements 

- Python >= 3.9
- Cython
- pkg-config
- openblas 
- gcc_multi 
- glibc_multi 
- gccMultiStdenv
- ninja
- fortran
- meson
- Cuda nvcc
- zlib
- uv

Further dependencies will be installed during the setup and installation process.

## Setup 

There is a known bug in the flash-atten python module at time of writing.  This bug may require manual intervention in order to circumvent.

```bash 
git clone https://github.com/BIDS-Xu-Lab/Kiwi-LLaMA.git
cd Kiwi-LLaMA
uv sync
```

In the event there is trouble identifying all the needed system packages, additional information is available in the `flake.nix` file.

## Use

### Training 

To instruction tune the model from scratch, you will have to specify your huggingface token in train.py, and run it by:
```bash
### set CUDA_VISIBLE_DEVICES to multiple GPU for multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py
```

### Inference 

For inference, specify the model directory in inference.py and run it by:
```bash
python kiwillama/inference.py
```

### Data

For the MTSample dataset, we are providing both entity offsets and texts, as well as the instruction tuning format. The MIMIC-III and 2010 i2b2 datasets, we are only providing entity offsets and texts without the original notes.

Originally, weights had to be obtained through [https://kiwi.clinicalnlp.org/download/](https://kiwi.clinicalnlp.org/download/), but they have since been added in `data/`. 


## Citation

```bibtex
@article{hu2024information,
  title={Information Extraction from Clinical Notes: Are We Ready to Switch to Large Language Models?},
  author={Hu, Yan and Zuo, Xu and Zhou, Yujia and Peng, Xueqing and Huang, Jimin and Keloth, Vipina K and Zhang, Vincent J and Weng, Ruey-Ling and Chen, Qingyu and Jiang, Xiaoqian and others},
  journal={arXiv preprint arXiv:2411.10020},
  year={2024}
}
```
