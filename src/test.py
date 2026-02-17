import os
import torch
from seq_encoder import build_dataset
from gnn_models import CustomGAT
from pipeline import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Testing on D. melanogaster -----------
test_data_dir = os.path.join("..", "data")
test_genes_path = os.path.join(test_data_dir, "melanogaster_genes.fasta")
test_labels_path = os.path.join(test_data_dir, "melanogaster_labels.txt")

models_dir = os.path.join("..", "models")
model_path = os.path.join(models_dir, "model.pt")
vocab_path = os.path.join(models_dir, "vocab.pth")
config_path = os.path.join(models_dir, "config.pth")

# Reload train vocab
vocab = torch.load(vocab_path)
test_graphs, _ = build_dataset(test_genes_path, test_labels_path, k=3)

# Recreate SAME model architecture similar to train
config = torch.load(config_path)
test_model = CustomGAT(**config).to(device)

# Load saved weights
test_model.load_state_dict(torch.load(model_path, map_location=device))

results = test(
    graphs=test_graphs,
    model=test_model,
    model_path=model_path,
    batch_size=len(test_graphs)//2,
    device=device,
    return_dataframe=True
)