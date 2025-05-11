# =====================
# File: main.py
# =====================
import torch
from setup.setup_env import setup_environment
from src.data_loading import load_and_preprocess_data, index_columns
from src.graph_builder import build_heterograph, assign_labels_to_graph
from src.features import get_business_features, get_review_features, get_reviewer_features
from src.models import RGCN, LabelSmoothing
from src.train_eval import train, evaluate, extract_embeddings
import numpy as np
import pandas as pd

# --- Setup ---
device = setup_environment()

# --- Load Data ---
data_path = '/content/drive/MyDrive/RGCN Classification/data/real_data.csv'
df = load_and_preprocess_data(data_path)
df, _, _, _ = index_columns(df)

# --- Build Graph ---
hg = build_heterograph(df)
hg = assign_labels_to_graph(hg, df)
hg = hg.to(device)

# --- Add Node Features ---
X_biz = torch.tensor(get_business_features(df)).float()
X_rev = torch.tensor(get_review_features(df)).float()
X_reviewer = torch.tensor(get_reviewer_features(df)).float()

hg.nodes['bizIdx'].data['feat'] = X_biz.to(device)
hg.nodes['revIdx'].data['feat'] = X_rev.to(device)
hg.nodes['reviewerIdx'].data['feat'] = X_reviewer.to(device)

# --- Hyperparameters ---
labels = hg.nodes['bizIdx'].data['label'].cpu().numpy()
bn = True
self_loop = True
num_layers_list = [3]
batch_sizes = [128]
h_feats_list = [64]
dropout_list = [0.5]
epoch_list = [2]
es_criteria_list = [20]
sampler_type_list = [2]
n_neighbors_list = [3]
smoothing_coeff_list = [0]
all_results = []

for num_layers in num_layers_list:
    for batch_size in batch_sizes:
        for h_feats in h_feats_list:
            for dropout in dropout_list:
                for epochs in epoch_list:
                    for es_criteria in es_criteria_list:
                        for sampler_type in sampler_type_list:
                            n_neighbors_range = n_neighbors_list if sampler_type == 2 else [None]
                            for n_neighbors in n_neighbors_range:
                                for smoothing_coeff in smoothing_coeff_list:
                                    print(f"Running with num_layers={num_layers}, batch_size={batch_size}, h_feats={h_feats}, dropout={dropout}, \
                                          epochs={epochs}, es_criteria={es_criteria}, sampler_type={sampler_type}, n_neighbors={n_neighbors}, smoothing_coeff={smoothing_coeff}")

                                    model_kwargs = dict(
                                        in_feats=X_biz.shape[1], h_feats=h_feats, num_classes=2, dropout=dropout,
                                        num_bases=None, self_loop=self_loop, bn=bn, num_layers=num_layers,
                                    )
                                    model = RGCN(hg, **model_kwargs).to(device)

                                    loss_fn = LabelSmoothing(smoothing=smoothing_coeff)
                                    weight_for_class_1 = sum(labels) / len(labels)
                                    weight_for_class_0 = 1 - weight_for_class_1
                                    weight_values = [weight_for_class_0, weight_for_class_1]

                                    rand_perm = np.random.permutation(len(labels))
                                    train_idx = rand_perm[:int(0.8 * len(labels))]
                                    val_idx = rand_perm[int(0.8 * len(labels)):]
                                    train_mask = torch.zeros(len(labels), dtype=torch.bool); train_mask[train_idx] = True
                                    val_mask = torch.zeros(len(labels), dtype=torch.bool); val_mask[val_idx] = True
                                    split_mask = {'train': train_mask.to(device), 'valid': val_mask.to(device), 'test': val_mask.to(device)}

                                    model_name = f"model_{smoothing_coeff}_{num_layers}_{batch_size}_{h_feats}_{dropout}_{epochs}_{es_criteria}_{sampler_type}_{n_neighbors}_bn{bn}_self{self_loop}.pt"

                                    train_losses, val_losses = train(
                                        hg.to(device), split_mask, model.to(device), epochs, batch_size, sampler_type, n_neighbors, weight_values, None,
                                        device, save_path=model_name, loss_fn=loss_fn, lr=0.001, es_criteria=es_criteria, verbose=True, weight=True)

                                    metrics = evaluate(model.to(device), hg.to(device), split_mask['valid'].to(device), batch_size, sampler_type, n_neighbors, best_model_fp=model_name)

                                    # --- Extract Embeddings ---
                                    train_embeddings = extract_embeddings(
                                        model.to(device), hg.to(device), mask=split_mask['train'].to(device),
                                        batch_size=batch_size, sampler_type=sampler_type, n_neighbors=n_neighbors,
                                        best_model_fp=model_name
                                    )
                                    train_labels = labels[train_mask.cpu().numpy()]

                                    val_embeddings = extract_embeddings(
                                        model.to(device), hg.to(device), mask=split_mask['valid'].to(device),
                                        batch_size=batch_size, sampler_type=sampler_type, n_neighbors=n_neighbors,
                                        best_model_fp=model_name
                                    )
                                    val_labels = labels[val_mask.cpu().numpy()]

                                    # Save embeddings (optional)
                                    pd.DataFrame(val_embeddings).assign(label=val_labels).to_csv(
                                        f"/content/drive/MyDrive/RGCN Classification/val_embeddings_{model_name}.csv", index=False)

                                    all_results.append({
                                        'num_layers': num_layers,
                                        'batch_size': batch_size,
                                        'h_feats': h_feats,
                                        'dropout': dropout,
                                        'epochs': epochs,
                                        'es_criteria': es_criteria,
                                        'sampler_type': sampler_type,
                                        'n_neighbors': n_neighbors,
                                        'smoothing_coeff': smoothing_coeff,
                                        'accuracy': metrics[0],
                                        'precision': metrics[1],
                                        'recall': metrics[2],
                                        'f1': metrics[3],
                                        'auc': metrics[5]
                                    })

# --- Save Results ---
df_results = pd.DataFrame(all_results)
df_results.to_csv('/content/drive/MyDrive/RGCN Classification/CO_github_post_code_results.csv', index=False)
print("All runs completed. Results saved.")
