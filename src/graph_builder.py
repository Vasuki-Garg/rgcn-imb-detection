# =====================
# File: src/graph_builder.py
# =====================
import torch
import dgl


def build_heterograph(df):
    edge_dict = dict()
    edges = [('bizIdx', 'revIdx'), ('revIdx', 'reviewerIdx')]
    for src_type, dst_type in edges:
        edge_name = f'{src_type}-{dst_type}'
        fwd = (src_type, edge_name, dst_type)
        bwd = (dst_type, f'{dst_type}-{src_type}', src_type)
        edge_pairs = df[[src_type, dst_type]].drop_duplicates().values
        src = edge_pairs[:, 0]
        dst = edge_pairs[:, 1]
        edge_dict[fwd] = (torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64))
        edge_dict[bwd] = (torch.tensor(dst, dtype=torch.int64), torch.tensor(src, dtype=torch.int64))
    return dgl.heterograph(edge_dict)


def assign_labels_to_graph(hg, df):
    biz_labels = df.drop_duplicates(subset='bizIdx').sort_values('bizIdx')['label'].values
    hg.nodes['bizIdx'].data['label'] = torch.tensor(biz_labels, dtype=torch.int64)
    hg.nodes['revIdx'].data['label'] = torch.zeros(hg.number_of_nodes('revIdx'))
    hg.nodes['reviewerIdx'].data['label'] = torch.zeros(hg.number_of_nodes('reviewerIdx'))
    return hg
