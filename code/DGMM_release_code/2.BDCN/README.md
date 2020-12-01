# Predict Edge probability map in historical map by BDCN

## Prerequisites

```python
numpy==1.19.4
torch>=1.7.0
```

- install requirement.txt into your virtualenv

```bash
pip install -r requirement.txt
```

- git clone 
```bash
git clone https://github.com/soduco/paper-dgmm2021.git
```

- Download the historical map dataset in release and put it into folder ./2.BDCN/historical_map_data/

- Download the historical map pre-train model in release and put it into ./2.BDCN/params/

## Training (Validating) and Testing

- Training BDCN with learning rate 5 * 10^-5
```python
python train.py --cuda --lr 5e-5
```

- Inferencing BDCN: The results will be saved in the default folder ./2.BDCN/results/
```python
python test.py --cuda --model ./2.BDCN/params/historical_map_pretain_model.pth
```

