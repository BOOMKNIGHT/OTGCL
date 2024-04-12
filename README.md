# OTGCL
The official implementation for "Structural Invariant Augumentation for Graph Contrastive Learning based on Optimal Transport"(OTGCL).	

## Dependencies

- Python 3.8.10
- PyTorch 1.10.0
- torch_geometric 2.1.0
- pyGCL 0.1.0
- POT 0.9.1

## Training & Evaluation

Node classification:

```
python train.py --dataset cora --config config1.yaml --task node_classification
```

Degree prediction:

```
python train.py --dataset cora --config config2.yaml --task degree_prediction
```





