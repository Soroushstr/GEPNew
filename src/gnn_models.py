import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GraphSAGE(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean"
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        # ----- GraphSAGE layers -----
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        # ----- Global pooling -----
        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)   # binary classification
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index

        # ---- GraphSAGE layers (NO edge weights allowed) ----
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)

        # ---- Classifier ----
        out = self.mlp(g)
        return out

class WeightedGCN(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            emb_dim=128, 
            hidden_dim=128, 
            num_layers=2, 
            dropout=0.2,
            pool="mean"
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        # ----- GCN layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            self.convs.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
            
        # ----- Global pooling -----
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError("pool must be 'mean', 'max', or 'sum'.")
        
        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)

        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1)

        # ---- GCN layers        
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        # ---- Global pooling ----
        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)
        
        # ---- Classifier ----        
        out = self.mlp(g)
        return out
    
class DefaultGINModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        emb_dim=128, 
        hidden_dim=128, 
        num_layers=2, 
        dropout=0.2,
        pool="mean"
    ):
        super(DefaultGINModel, self).__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        # ----- GIN layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            in_dim = hidden_dim

        # ----- Global pooling -----
        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]
        
        # ----- Final MLP -----        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        # ---- Node embedding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index

        # ---- GIN layers  
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        g = self.pool(x, batch)

        # ---- Classifier ----  
        out = self.mlp(g)
        return out

class WeightedGINConv(MessagePassing):
    def __init__(self, mlp, eps=0.0, train_eps=False):
        super().__init__(aggr="add")
        self.mlp = mlp

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_weight is not None:
            edge_weight = torch.cat(
                [edge_weight, torch.ones(x.size(0), device=x.device)]
            )

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = (1 + self.eps) * x + out
        return self.mlp(out)

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

class WeightedGIN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean",
        train_eps=True
    ):
        super().__init__()

        # ----- Embedding -----
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        # ----- Weighted GIN layers -----
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(input_dim))
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(
                WeightedGINConv(mlp, train_eps=train_eps)
            )
            input_dim = hidden_dim

        # ----- Global pooling -----
        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        # ----- Final MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x = self.embedding(data.x.view(-1))
        edge_index = data.edge_index

        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        g = self.pool(x, batch)
        return self.mlp(g)

class EdgeAttrGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, negative_slope=0.2, dropout=0.2):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        self.att_src = nn.Parameter(torch.Tensor(out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(out_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)
            edge_attr = torch.cat(
                [edge_attr, torch.ones(x.size(0), device=x.device)]
            )

        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_attr=edge_attr
        )

    def message(self, x_i, x_j, edge_attr, index):
        alpha = (x_i * self.att_dst).sum(dim=-1) + \
                (x_j * self.att_src).sum(dim=-1)

        alpha = self.leaky_relu(alpha)

        if edge_attr is not None:
            alpha = alpha * edge_attr

        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, 1)

class CustomGAT(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean"
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(in_dim))
            self.convs.append(
                EdgeAttrGATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    dropout=dropout
                )
            )
            in_dim = hidden_dim

        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x = self.embedding(data.x.view(-1))
        edge_index = data.edge_index

        edge_attr = getattr(data, "edge_attr", None)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        g = self.pool(x, batch)
        return self.mlp(g)


class AttentiveWeightedGINConv(MessagePassing):
    def __init__(self, mlp, eps=0.0, train_eps=False, dropout=0.2):
        super().__init__(aggr="add")

        self.mlp = mlp
        self.dropout = dropout

        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))

        # Single-head attention (SAFE)
        hidden_dim = mlp[0].in_features
        self.att = nn.Linear(2 * hidden_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_weight is not None:
            edge_weight = torch.cat(
                [edge_weight, torch.ones(x.size(0), device=x.device)]
            )

        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_weight=edge_weight
        )

        out = (1 + self.eps) * x + out
        return self.mlp(out)
    
    def message(self, x_i, x_j, edge_weight, index):
        # Compute attention
        alpha = torch.cat([x_i, x_j], dim=-1)
        alpha = self.att(alpha)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        return alpha * x_j

class AttentiveWeightedGIN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        pool="mean",
        train_eps=True
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        input_dim = emb_dim
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(input_dim))
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(
                AttentiveWeightedGINConv(
                    mlp,
                    train_eps=train_eps,
                    dropout=dropout
                )
            )
            input_dim = hidden_dim

        self.pool = {
            "mean": global_mean_pool,
            "max": global_max_pool,
            "sum": global_add_pool
        }[pool]

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x = self.embedding(data.x.view(-1))
        edge_index = data.edge_index

        edge_weight = None
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_weight = data.edge_attr.view(-1)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch = getattr(data, "batch",
                        torch.zeros(x.size(0), dtype=torch.long, device=x.device))

        g = self.pool(x, batch)
        return self.mlp(g)

