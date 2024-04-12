# Instruction-tuned_Large_Language_Models_are_Superior_Clinical_Information_Extractors

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
