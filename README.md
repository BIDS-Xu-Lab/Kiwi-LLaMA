# Information Extraction from Clinical Notes: Are We Ready for Adopting Large Language Models?
## Overview
This repository contains the code and resources for Kiwi, a LLM-based clinical Information Extraction (IE) system designed to extract medical problem, treatment, test, and drug entities, and their modifiers from clinical note.

### Data
For the MTSample dataset, we are providing both entity offsets and texts, as well as the instruction tuning format. The MIMIC-III and 2010 i2b2 datasets, we are only providing entity offsets and texts without the original notes.

## Environment
```bash
git clone https://github.com/BIDS-Xu-Lab/Kiwi-LLaMA.git
cd Kiwi-LLaMA
pip install -r requirements.txt
```

## Instruction Tuning
To instruction tune the model from scratch, you will have to specify your huggingface token in train.py, and run it by:
```bash
### set CUDA_VISIBLE_DEVICES to multiple GPU for multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py
```

## Inference
For inference, you will have to acquire model weights from [https://kiwi.clinicalnlp.org/download/](https://kiwi.clinicalnlp.org/download/). Then specify the model directory in inference.py, and run it by:
```bash
python inference.py
```

## Citation
```bibtex
@article{hu2024information,
  title={Information Extraction from Clinical Notes: Are We Ready to Switch to Large Language Models?},
  author={Hu, Yan and Zuo, Xu and Zhou, Yujia and Peng, Xueqing and Huang, Jimin and Keloth, Vipina K and Zhang, Vincent J and Weng, Ruey-Ling and Chen, Qingyu and Jiang, Xiaoqian and others},
  journal={arXiv preprint arXiv:2411.10020},
  year={2024}
}
```
