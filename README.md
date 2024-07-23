# Information Extraction from Clinical Notes: Are We Ready for Adopting Large Language Models?

## Environment
```bash
git clone https://github.com/BIDS-Xu-Lab/MedLey-LLaMA.git
cd MedLey-LLaMA
pip install -r requirements.txt
```

## Instruction Tuning
```bash
### change CUDA_VISIBLE_DEVICES to multiple GPU for multi-GPU training
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py
```

## Inference
```bash
python inference.py
```
