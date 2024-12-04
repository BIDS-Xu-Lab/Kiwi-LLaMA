# Information Extraction from Clinical Notes: Are We Ready for Adopting Large Language Models?
## Overview
This repository contains the code and resources for Kiwi, a LLM-based clinical Information Extraction (IE) system designed to extract medical problem, treatment, test, and drug entities from clinical note.

### Data
For the MTSample dataset, we are providing both entity offsets and texts, as well as the instruction tuning format. The MIMIC-III and 2010 i2b2 datasets, we are only providing entity offsets and texts without the original notes.

## Environment
```bash
git clone https://github.com/BIDS-Xu-Lab/Kiwi-LLaMA.git
cd Kiwi-LLaMA
pip install -r requirements.txt
```

## Instruction Tuning
```bash
### set CUDA_VISIBLE_DEVICES to multiple GPU for multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py
```

## Inference
```bash
python inference.py
```
