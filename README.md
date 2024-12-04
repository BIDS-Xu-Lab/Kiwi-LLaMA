# Information Extraction from Clinical Notes: Are We Ready for Adopting Large Language Models?

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
