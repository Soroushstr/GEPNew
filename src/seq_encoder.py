import torch
from torch_geometric.data import Data
from collections import Counter

def read_fasta(filepath):
    records = []
    with open(filepath, 'r') as f:
        header, seq_lines = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq_lines).upper()))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            records.append((header, ''.join(seq_lines).upper()))
    return records

def read_labels(filepath):
    with open(filepath, 'r') as f:
        return [int(l.strip()) for l in f if l.strip()]

def build_vocab(seqs, k, min_count=1):
    cnt = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if set(kmer) <= set('ATCGN'):
                cnt[kmer] += 1
    items = sorted([kmer for kmer, c in cnt.items() if c >= min_count])
    vocab = {"<UNK>": 0}
    for i, kmer in enumerate(items, start=1):
        vocab[kmer] = i
    return vocab

def seq_to_graph(seq, k, vocab, bidirectional=False, normalize=True):
    kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
    if len(kmers) == 0:
        return None
    unk = vocab.get('<UNK>', 0)

    unique_kmers = list(dict.fromkeys(kmers))
    node_index_map = {kmer: idx for idx, kmer in enumerate(unique_kmers)}

    node_ids = [vocab.get(kmer, unk) for kmer in unique_kmers]
    x = torch.tensor(node_ids, dtype=torch.long).unsqueeze(1)

    edge_counter = Counter()
    for i in range(len(kmers) - 1):
        src = node_index_map[kmers[i]]
        dst = node_index_map[kmers[i + 1]]
        edge_counter[(src, dst)] += 1
        if bidirectional:
            edge_counter[(dst, src)] += 1

    edges, weights = zip(*edge_counter.items())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)

    if normalize:
        row, col = edge_index
        out_degree = torch.zeros(len(unique_kmers), dtype=torch.float)
        for (src, _), w in zip(edges, weights):
            out_degree[src] += w

        norm_weights = []
        for (src, _), w in zip(edges, weights):
            if out_degree[src] > 0:
                norm_weights.append(w / out_degree[src])
            else:
                norm_weights.append(0.0)
        edge_attr = torch.tensor(norm_weights, dtype=torch.float)

    edge_attr = edge_attr.unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.kmers = unique_kmers
    return data


def build_dataset(fasta_path, labels_path, k, vocab=None):
    print("########## Start building garphs from seqs...")

    records = read_fasta(fasta_path)
    labels = read_labels(labels_path)

    seqs = [r[1] for r in records]
    gene_ids = [r[0] for r in records]

    if vocab is None:
        vocab = build_vocab(seqs, k)

    graphs = []
    for seq, lab, gid in zip(seqs, labels, gene_ids):
        g = seq_to_graph(seq, k, vocab)
        if g is None:
            continue

        g.y = torch.tensor([lab], dtype=torch.long)
        g.gene_id = gid
        graphs.append(g)

    print("########## Graph Built!")
    return graphs, vocab
