import os
import torch
from seq_encoder import build_dataset
from gnn_models import CustomGAT 
from pipeline import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Training on C. elegans -----------
data_dir = os.path.join("..", "data")
models_dir = os.path.join("..", "models")

genes_path = os.path.join(data_dir, "elegans_genes.fasta")
labels_path = os.path.join(data_dir, "elegans_labels.txt")
model_path = os.path.join(models_dir, "model.pt")
vocab_path = os.path.join(models_dir, "vocab.pth")
config_path = os.path.join(models_dir, "config.pth")

graphs, vocab = build_dataset(genes_path, labels_path, k=3)
torch.save(vocab, vocab_path)

config = {
    "vocab_size": len(vocab),
    "emb_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "pool": "mean",
}
torch.save(config, config_path)

model = CustomGAT(**config).to(device)

results = train(
            graphs,
            model,
            batch_size=64,
            epoch_n=100,
            learning_rate=1e-3,
            weighted_sampling=True,
            use_scheduler=True,
            scheduler_patience=3,
            scheduler_factor=0.5,
            use_gradient_clipping=True,
            clip_value=1.0,
            model_path=model_path,
            device=device,
            val_split=0.2,
            random_seed=111,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001
)