# GEP-GNN: Gene Essentiality Prediction with Graph Neural Networks

- This repository implements a pipeline for **gene essentiality prediction** using **Graph Neural Networks (GNNs)** between species.  
- Gene sequences are represented as *k-mer graphs* using De-Bruijn technique, where edges denote transitions between k-mers, optionally weighted by transition probabilities for denoting frequencies of each edge.  
- The system supports multiple GNN architectures — **Unweighted GrapSAGE**, **Weighted GCN**, **Edge-Aware GAT**, **Weighted GIN**, with different **Global Pooling** strategies as well as **Hierarchical Pooling**.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/GEP-GNN.git
cd GEP-GNN

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

1. Build a Graph Dataset

```python
from seq_encoder import build_dataset

graphs, vocab = build_dataset(genes_path, labels_path, k=3)
```

2. Train a Model

```python
from gnn_models import CustomGAT 
from pipeline import train

model = CustomGAT(**config).to(device)
results = train(graphs, model)
```

3. Test on Another Species

```python
from gnn_models import CustomGAT
from pipeline import test

test_graphs, _ = build_dataset(test_genes_path, test_labels_path, k=3)

test_model = CustomGAT(**config).to(device)
results = test(graphs=test_graphs, model=test_model)
```

---

## Modules Overview

- seq_encoder.py
  - Converts FASTA sequences into k-mer transition graphs
  - Supports:
    - Weighted edges (transition frequencies)
    - Normalized adjacency matrices
    - Node degree and sequence length features
    - Outputs PyTorch Geometric Data objects

- gnn_models.py
  - Defines multiple architectures:
    - Unweighted GraphSAGE – baseline GraphSAGE without weighted edges
    - Weighted GCN – Graph Convolution Neural Networks with weighted edges
    - EdgeAttr GAT – edge-aware attention mechanism
    - GINModel – expressive node embedding aggregation
    - Global Pooling - mean, max, and sum over node features
    - DiffPool – hierarchical pooling for graph-level embedding

- pipeline.py
  - Unified training and testing functions
  - Handles learning rate scheduling, checkpointing, and GPU detection
  - Logs loss and metric curves per epoch
  - Supports cross-species evaluation

---

## License

This project is released under the MIT License — feel free to use and modify with attribution.

---

⭐ If you find this repository useful, please consider starring it on GitHub!

