# Instruction-tuned_Large_Language_Models_are_Superior_Clinical_Information_Extractors

## Environment
```bash
git clone https://github.com/BIDS-Xu-Lab/Instruction-tuned_Large_Language_Models_are_Superior_Clinical_Information_Extractors.git
cd Instruction-tuned_Large_Language_Models_are_Superior_Clinical_Information_Extractors
pip install -r requirements.txt
```

## Instruction Tuning
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py
```

## Inference
```bash
python inference.py
```