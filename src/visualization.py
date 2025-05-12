# =====================
# File: src/visualization.py
# =====================
import torch
import matplotlib.pyplot as plt
import networkx as nx
import dgl
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.manifold import TSNE


def graph_hg(hg_tmp, **kwargs):
    hg_label = hg_tmp.ndata['label']['bizIdx'].int()
    g = dgl.to_homogeneous(hg_tmp)
    g.ndata['ntype'] = g.ndata['_TYPE']
    homogeneous_ntypes = g.ndata['ntype']
    hetero_ids = hg_tmp.ndata['_ID']

    business_mask = (homogeneous_ntypes == 0)
    review_mask = (homogeneous_ntypes == 1)
    reviewer_mask = (homogeneous_ntypes == 2)

    homogeneous_ids_tensor = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    homogeneous_ids_tensor[business_mask] = hetero_ids['bizIdx']
    homogeneous_ids_tensor[review_mask] = hetero_ids['revIdx']
    homogeneous_ids_tensor[reviewer_mask] = hetero_ids['reviewerIdx']
    g.ndata['iden'] = homogeneous_ids_tensor

    nx_graph = g.to_networkx(node_attrs=['ntype', 'iden']).to_undirected()
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(nx_graph, seed=40)

    node_styles = {
        'bizIdx': {'shape': 'o'},
        'revIdx': {'shape': 's', 'color': 'palegreen'},
        'reviewerIdx': {'shape': '^', 'color': 'bisque'}
    }

    for idx, ntype in enumerate(hg_tmp.ntypes):
        nodes = [n for n, data in nx_graph.nodes(data=True) if hg_tmp.ntypes[data['ntype']] == ntype]
        labels = {n: f"{ntype}\n{data['iden'].item()}" for n, data in nx_graph.nodes(data=True) if hg_tmp.ntypes[data['ntype']] == ntype}

        shape = node_styles[ntype]['shape']
        if ntype == 'bizIdx':
            node_colors = ['powderblue' if v == 0 else 'salmon' if v == 1 else 'black' for v in hg_label]
        else:
            node_colors = node_styles[ntype].get('color')

        nx.draw_networkx_nodes(nx_graph, pos, nodelist=nodes, node_size=800, node_color=node_colors, label=ntype, node_shape=shape)
        nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=12)

    nx.draw_networkx_edges(nx_graph, pos, alpha=0.5)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='powderblue', markersize=7, label='Business (Label 0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=7, label='Business (Label 1)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='palegreen', markersize=7, label='Review'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='bisque', markersize=7, label='Reviewer')
    ]

    plt.legend(handles=legend_handles, loc="upper left", fontsize=18, markerscale=8, frameon=True, fancybox=True, framealpha=1, borderpad=2, labelspacing=3, handletextpad=2)
    plt.savefig('/content/drive/MyDrive/RGCN_IMB_Detection/sub_graph_network.png')
    plt.show()


def plot_neighborhood(hg, dataloader, N_plots=5):
    for i, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        if i >= N_plots:
            break
        hg_tmp = dgl.node_subgraph(hg, input_nodes)
        graph_hg(hg_tmp)


def plot_tsne(embeddings, labels, title="t-SNE Visualization of Test Embeddings"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    cmap = ListedColormap(["#ff0000", "#0000ff"])
    norm = BoundaryNorm([0, 0.5, 1], cmap.N)
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap=cmap, norm=norm, alpha=0.7)

    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_label("Class Labels", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Class 0", "Class 1"])

    plt.xlabel("t-SNE Component 1", fontsize=22)
    plt.ylabel("t-SNE Component 2", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.savefig('/content/drive/MyDrive/RGCN_IMB_Detection/t_sne.png')
    plt.show()
