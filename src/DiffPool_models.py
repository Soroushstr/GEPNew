import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

class WeightedGINDense(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True):
        super().__init__()
        self.mlp = mlp
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, adj):
        # adj: [B, N, N], x: [B, N, F]
        I = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj_with_self = adj + I
        deg = adj_with_self.sum(-1, keepdim=True)
        out = torch.bmm(adj_with_self, x) / (deg + 1e-6)
        out = (1 + self.eps) * x + out
        return self.mlp(out)


class WeightedGINDiffPool(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128,
                 cluster_ratio1=0.25, cluster_ratio2=0.10,
                 num_classes=2, dropout=0.3):
        super(WeightedGINDiffPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cluster_ratio1 = cluster_ratio1
        self.cluster_ratio2 = cluster_ratio2

        # -------- Shared MLP Builder --------
        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

        # -------- Block 1 (Before 1st DiffPool) --------
        self.gnn_embed1 = WeightedGINDense(make_mlp(emb_dim, hidden_dim))
        self.gnn_embed2 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_assign1 = WeightedGINDense(make_mlp(emb_dim, hidden_dim))

        self.norm1 = nn.LayerNorm(hidden_dim)

        # -------- Block 2 (After 1st DiffPool) --------
        self.gnn_embed3 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed4 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed5 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_assign2 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))

        self.norm2 = nn.LayerNorm(hidden_dim)

        # -------- Block 3 (After 2nd DiffPool) --------
        self.gnn_embed6 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed7 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))
        self.gnn_embed8 = WeightedGINDense(make_mlp(hidden_dim, hidden_dim))

        self.norm3 = nn.LayerNorm(hidden_dim)

        # -------- Readout --------
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch, edge_attr)

        # Dynamically determine cluster sizes
        N_max = x_dense.size(1)
        assign_dim1 = max(2, int(self.cluster_ratio1 * N_max))
        assign_dim2 = max(2, int(self.cluster_ratio2 * assign_dim1))

        # ========== Block 1 ==========
        z = F.relu(self.gnn_embed1(x_dense, adj_dense))
        z = F.relu(self.gnn_embed2(z, adj_dense))
        s = F.softmax(self.gnn_assign1(x_dense, adj_dense)[..., :assign_dim1], dim=-1)
        z, adj, l1, e1 = dense_diff_pool(z, adj_dense, s, mask)

        z = self.norm1(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # ========== Block 2 ==========
        z = F.relu(self.gnn_embed3(z, adj))
        z = F.relu(self.gnn_embed4(z, adj))
        z = F.relu(self.gnn_embed5(z, adj))
        s = F.softmax(self.gnn_assign2(z, adj)[..., :assign_dim2], dim=-1)
        z, adj, l2, e2 = dense_diff_pool(z, adj, s)

        z = self.norm2(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # ========== Block 3 ==========
        z = F.relu(self.gnn_embed6(z, adj))
        z = F.relu(self.gnn_embed7(z, adj))
        z = F.relu(self.gnn_embed8(z, adj))
        z = self.norm3(z)

        # Graph-level readout (mean over remaining nodes)
        g = z.mean(dim=1)
        out = self.mlp_out(g)
        return out, (l1 + l2 + e1 + e2)

    
class SimpleSAGEDiffPool(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64,
                 cluster_ratio1=0.5, cluster_ratio2=0.25,
                 num_classes=2, dropout=0.2):
        super(SimpleSAGEDiffPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        
        # cluster ratios
        self.cluster_ratio1 = cluster_ratio1
        self.cluster_ratio2 = cluster_ratio2

        # Block 1: Pre-pooling
        self.embed1 = DenseSAGEConv(emb_dim, hidden_dim)
        self.assign1 = DenseSAGEConv(emb_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)  # BatchNorm often helps convergence

        # Block 2: After 1st pooling
        self.embed2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.assign2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        # Block 3: After 2nd pooling (final)
        self.embed3 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)

        # Simplified readout
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index, edge_attr = data.edge_index, data.edge_attr

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_dense, mask = to_dense_batch(x, batch)
        
        # Handle adjacency matrix
        if edge_attr is not None:
            adj_dense = to_dense_adj(edge_index, batch, edge_attr)
            if adj_dense.dim() == 4:
                adj_dense = adj_dense.squeeze(-1)
        else:
            adj_dense = to_dense_adj(edge_index, batch)

        # Dynamic cluster sizes
        N_max = x_dense.size(1)
        assign_dim1 = max(2, int(self.cluster_ratio1 * N_max))
        assign_dim2 = max(2, int(self.cluster_ratio2 * assign_dim1))

        # ========== Block 1 ==========
        h = F.relu(self.embed1(x_dense, adj_dense))
        s = F.softmax(self.assign1(x_dense, adj_dense)[..., :assign_dim1], dim=-1)
        h, adj, l1, e1 = dense_diff_pool(h, adj_dense, s, mask)
        
        # Apply norm - reshape for BatchNorm1d
        h_flat = h.view(-1, h.size(-1))
        h = self.norm1(h_flat).view(h.shape)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ========== Block 2 ==========
        h = F.relu(self.embed2(h, adj))
        s = F.softmax(self.assign2(h, adj)[..., :assign_dim2], dim=-1)
        h, adj, l2, e2 = dense_diff_pool(h, adj, s)
        
        h_flat = h.view(-1, h.size(-1))
        h = self.norm2(h_flat).view(h.shape)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ========== Block 3 ==========
        h = F.relu(self.embed3(h, adj))
        h_flat = h.view(-1, h.size(-1))
        h = self.norm3(h_flat).view(h.shape)

        # Global mean pooling
        g = h.mean(dim=1)
        out = self.mlp_out(g)
        
        return out, (e1 + e2)
    

class SimpleSAGENoPool(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64,
                 pool_ratio1=0.5, pool_ratio2=0.25,
                 num_classes=2, dropout=0.2):
        super(SimpleSAGENoPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        
        # pool ratios (now used for potential sampling, but we'll use global pooling)
        self.pool_ratio1 = pool_ratio1
        self.pool_ratio2 = pool_ratio2

        # Block 1: First SAGE layer
        self.sage1 = DenseSAGEConv(emb_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)

        # Block 2: Second SAGE layer  
        self.sage2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)

        # Block 3: Third SAGE layer (final)
        self.sage3 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)

        # Simplified readout
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index, edge_attr = data.edge_index, data.edge_attr

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_dense, mask = to_dense_batch(x, batch)
        
        # Handle adjacency matrix
        if edge_attr is not None:
            adj_dense = to_dense_adj(edge_index, batch, edge_attr)
            if adj_dense.dim() == 4:
                adj_dense = adj_dense.squeeze(-1)
        else:
            adj_dense = to_dense_adj(edge_index, batch)

        # ========== Block 1 ==========
        h = F.relu(self.sage1(x_dense, adj_dense))
        h_flat = h.view(-1, h.size(-1))
        h = self.norm1(h_flat).view(h.shape)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ========== Block 2 ==========
        h = F.relu(self.sage2(h, adj_dense))
        h_flat = h.view(-1, h.size(-1))
        h = self.norm2(h_flat).view(h.shape)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ========== Block 3 ==========
        h = F.relu(self.sage3(h, adj_dense))
        h_flat = h.view(-1, h.size(-1))
        h = self.norm3(h_flat).view(h.shape)

        # Global mean pooling
        g = h.mean(dim=1)
        out = self.mlp_out(g)
        
        return out, torch.tensor(0.0, device=out.device)  # Return 0 for compatibility


class MinimalSAGEDiffPool(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=64,
                 cluster_ratio=0.5, num_classes=2, dropout=0.2):
        super(MinimalSAGEDiffPool, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = dropout
        self.cluster_ratio = cluster_ratio

        # Minimal architecture - just one pooling step
        self.pre_pool1 = DenseSAGEConv(emb_dim, hidden_dim)
        self.pre_pool2 = DenseSAGEConv(hidden_dim, hidden_dim)
        self.assign = DenseSAGEConv(emb_dim, hidden_dim)
        
        self.post_pool = DenseSAGEConv(hidden_dim, hidden_dim)
        
        self.norm = nn.BatchNorm1d(hidden_dim)
        
        # Simple classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)
        edge_index = data.edge_index

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_dense, mask = to_dense_batch(x, batch)
        adj_dense = to_dense_adj(edge_index, batch)
        
        if adj_dense.dim() == 4:
            adj_dense = adj_dense.squeeze(-1)

        # Single pooling step
        N_max = x_dense.size(1)
        assign_dim = max(2, int(self.cluster_ratio * N_max))

        # Pre-pool processing
        h = F.relu(self.pre_pool1(x_dense, adj_dense))
        h = F.relu(self.pre_pool2(h, adj_dense))
        
        # Pooling
        s = F.softmax(self.assign(x_dense, adj_dense)[..., :assign_dim], dim=-1)
        h, adj, link_loss, ent_loss = dense_diff_pool(h, adj_dense, s, mask)
        
        # Post-pool processing
        h = F.relu(self.post_pool(h, adj))
        
        # Normalization and readout
        h_flat = h.view(-1, h.size(-1))
        h = self.norm(h_flat).view(h.shape)
        g = h.mean(dim=1)
        
        out = self.classifier(g)
        return out, (link_loss + ent_loss)
    



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

class DenseSAGEBlock(nn.Module):
    """Dense GraphSAGE block for batched dense inputs (B, N, F) and (B, N, N)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = DenseSAGEConv(in_dim, out_dim)
        self.lin = nn.Linear(out_dim, out_dim)

    def forward(self, x, adj):
        x = self.conv(x, adj)
        x = F.relu(x)
        return self.lin(x)


class SAGEDiffPool(nn.Module):
    """
    GraphSAGE + DiffPool model with:
      - fixed precomputed cluster sizes (assign_dim1, assign_dim2)
      - LayerNorm between stages
      - dropout on assignment logits
      - returns (logits, aux_loss)
    """
    def __init__(self,
                 vocab_size,
                 emb_dim=64,
                 hidden_dim=64,
                 max_nodes=64,             # max nodes expected in graphs (for computing assign dims)
                 cluster_ratio1=0.25,
                 cluster_ratio2=0.25,
                 num_classes=2,
                 dropout=0.2,
                 assign_dropout=0.1,
                 use_layernorm=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.assign_dropout = assign_dropout
        self.use_layernorm = use_layernorm

        # compute fixed assign dims from max_nodes
        self.assign_dim1 = max(2, int(cluster_ratio1 * max_nodes))
        self.assign_dim2 = max(2, int(cluster_ratio2 * self.assign_dim1))

        # -------- Block 1 (pre pool) --------
        self.embed1 = DenseSAGEBlock(emb_dim, hidden_dim)
        self.embed2 = DenseSAGEBlock(hidden_dim, hidden_dim)
        self.assign1 = DenseSAGEBlock(emb_dim, self.assign_dim1)   # outputs assignment logits

        self.norm1 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # -------- Block 2 (between pools) --------
        self.embed3 = DenseSAGEBlock(hidden_dim, hidden_dim)
        self.embed4 = DenseSAGEBlock(hidden_dim, hidden_dim)
        self.embed5 = DenseSAGEBlock(hidden_dim, hidden_dim)
        self.assign2 = DenseSAGEBlock(hidden_dim, self.assign_dim2)

        self.norm2 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # -------- Block 3 (post pools) --------
        self.embed6 = DenseSAGEBlock(hidden_dim, hidden_dim)
        self.embed7 = DenseSAGEBlock(hidden_dim, hidden_dim)
        self.embed8 = DenseSAGEBlock(hidden_dim, hidden_dim)

        self.norm3 = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # -------- Readout --------
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        """
        data: a batched torch_geometric.data.Batch
        returns:
          logits: tensor (batch_size, num_classes)  -- RAW LOGITS (no softmax)
          aux_loss: scalar tensor (l1 + l2 + e1 + e2)
        """
        device = self.embedding.weight.device

        # embedding
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)                       # (total_nodes, emb_dim)

        batch = getattr(data, "batch", torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x_dense, mask = to_dense_batch(x, batch)        # x_dense: (B, N, F)
        # adjacency: account for edge_attr if present; to_dense_adj returns shape (B,N,N) or (B,N,N,1) if attr
        if getattr(data, "edge_attr", None) is not None:
            adj = to_dense_adj(data.edge_index, batch, data.edge_attr)
            # if adj has trailing dim for scalar attr, squeeze it
            if adj.dim() == 4 and adj.size(-1) == 1:
                adj = adj.squeeze(-1)
        else:
            adj = to_dense_adj(data.edge_index, batch)

        # ---------------- Block 1 ----------------
        h = self.embed1(x_dense, adj)
        h = self.embed2(h, adj)

        s_logits = self.assign1(x_dense, adj)             # (B, N, assign_dim1)
        if self.assign_dropout > 0:
            s_logits = F.dropout(s_logits, p=self.assign_dropout, training=self.training)
        s = F.softmax(s_logits, dim=-1)

        h, adj, l1, e1 = dense_diff_pool(h, adj, s, mask)   # h: (B, N1, hidden_dim)

        h = self.norm1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---------------- Block 2 ----------------
        h = self.embed3(h, adj)
        h = self.embed4(h, adj)
        h = self.embed5(h, adj)

        s_logits2 = self.assign2(h, adj)                   # (B, N1, assign_dim2)
        if self.assign_dropout > 0:
            s_logits2 = F.dropout(s_logits2, p=self.assign_dropout, training=self.training)
        s2 = F.softmax(s_logits2, dim=-1)

        h, adj, l2, e2 = dense_diff_pool(h, adj, s2)

        h = self.norm2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ---------------- Block 3 ----------------
        h = self.embed6(h, adj)
        h = self.embed7(h, adj)
        h = self.embed8(h, adj)

        h = self.norm3(h)

        # readout
        g = h.mean(dim=1)   # (B, hidden_dim)
        logits = self.mlp_out(g)

        aux = e1 + e2
        return logits, aux
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


# ============================================================
#  Custom Dense GraphSAGE with Edge Weights
# ============================================================
class WeightedDenseGraphSAGE(nn.Module):
    """
    Dense GraphSAGE layer with weighted edges.
    x:   [B, N, F]
    adj: [B, N, N]   (weights allowed)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adj):
        B, N, F = x.size()

        # Neighbor aggregation (weighted sum)
        neigh = torch.bmm(adj, x)  # [B, N, F]

        # Concatenate self + neighbors
        out = torch.cat([x, neigh], dim=-1)  # [B, N, 2F]

        return self.linear(out)


# ============================================================
#  Full GraphSAGE + DiffPool Model
# ============================================================
class WeightedSAGEDiffPool(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128,
                 cluster_ratio1=0.25, cluster_ratio2=0.10,
                 num_classes=2, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cluster_ratio1 = cluster_ratio1
        self.cluster_ratio2 = cluster_ratio2

        # -----------------------
        # BLOCK 1
        # -----------------------
        self.gnn_embed1 = WeightedDenseGraphSAGE(emb_dim, hidden_dim)
        self.gnn_embed2 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.gnn_assign1 = WeightedDenseGraphSAGE(emb_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # -----------------------
        # BLOCK 2
        # -----------------------
        self.gnn_embed3 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.gnn_embed4 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.gnn_embed5 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.gnn_assign2 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # -----------------------
        # BLOCK 3
        # -----------------------
        self.gnn_embed6 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.gnn_embed7 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.gnn_embed8 = WeightedDenseGraphSAGE(hidden_dim, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # -----------------------
        # Readout
        # -----------------------
        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    # ============================================================
    #  Forward Pass
    # ============================================================
    def forward(self, data):

        # ---- Dense Encoding ----
        x_idx = data.x.view(-1)
        x = self.embedding(x_idx)                      # [num_nodes, emb_dim]

        edge_index, edge_attr = data.edge_index, data.edge_attr
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1)

        batch = getattr(data, "batch", torch.zeros(x.size(0),
                       dtype=torch.long, device=x.device))

        x_dense, mask = to_dense_batch(x, batch)       # [B, N, F]
        adj_dense = to_dense_adj(edge_index, batch, edge_attr)  # [B, N, N]

        # ---- Dynamic cluster sizes ----
        N_max = x_dense.size(1)
        assign_dim1 = max(2, int(self.cluster_ratio1 * N_max))
        assign_dim2 = max(2, int(self.cluster_ratio2 * assign_dim1))

        # ============================================================
        #  BLOCK 1
        # ============================================================
        z = F.relu(self.gnn_embed1(x_dense, adj_dense))
        z = F.relu(self.gnn_embed2(z, adj_dense))

        s = self.gnn_assign1(x_dense, adj_dense)[..., :assign_dim1]
        s = F.softmax(s, dim=-1)

        z, adj, l1, e1 = dense_diff_pool(z, adj_dense, s, mask)

        z = self.norm1(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # ============================================================
        #  BLOCK 2
        # ============================================================
        z = F.relu(self.gnn_embed3(z, adj))
        z = F.relu(self.gnn_embed4(z, adj))
        z = F.relu(self.gnn_embed5(z, adj))

        s = self.gnn_assign2(z, adj)[..., :assign_dim2]
        s = F.softmax(s, dim=-1)

        z, adj, l2, e2 = dense_diff_pool(z, adj, s)

        z = self.norm2(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # ============================================================
        #  BLOCK 3
        # ============================================================
        z = F.relu(self.gnn_embed6(z, adj))
        z = F.relu(self.gnn_embed7(z, adj))
        z = F.relu(self.gnn_embed8(z, adj))

        z = self.norm3(z)

        # -------------------------
        # Global graph embedding
        # -------------------------
        g = z.mean(dim=1)   # [B, hidden_dim]
        out = self.mlp_out(g)
        return out, (l1 + l2 + e1 + e2)
