import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from mgcn import *
from utils_seurat import *
from gene2vec import load_gene_embeddings
import csv
from scipy.stats import pearsonr

torch.autograd.set_detect_anomaly(True)

def train(input_dir, dataset_labels, features_and_adjacencies_per_dataset, img_features_per_dataset, raw_gene_expression_data, gene_embeddings, gene_similarity_matrix,  mgcn_gnn_type, gene_gnn_type, mode, device):

    hidden_dims = [256, 512, 1024]
    output_dim = 1024
    alpha = 0.75
    input_dims = [features_and_adjacencies_per_dataset[dataset_labels[0]]['filtered_cell_level_features_tensor'].shape[1],
                  features_and_adjacencies_per_dataset[dataset_labels[0]]['filtered_patch_level_features_tensor'].shape[1],
                  2048
                  ]
    gene_embedding_input_dim = len(next(iter(gene_embeddings.values())))

    model_gene_embedding = GNN(gene_embedding_input_dim, output_dim, gene_gnn_type).to(device)
    model_mgcn = mGCN(input_dims, hidden_dims, output_dim, mgcn_gnn_type, alpha).to(device)

    optimizer = optim.AdamW([
        {'params': model_mgcn.parameters()},
        {'params': model_gene_embedding.parameters()}], lr=0.0001, weight_decay=1e-3)

    total_loss_values = []
    min_loss = float('inf')
    gene_embeddings_tensor = torch.tensor([gene_embeddings[gene] for gene in analyzed_genes], dtype=torch.float32).to(device)
    gene_similarity_matrix_tensor = process_gene_similarity_matrix(analyzed_genes, gene_similarity_matrix, device)
    gene_similarity_matrix_tensor_dgl = convert_adj_matrix_to_dgl_graph(gene_similarity_matrix_tensor, gene_gnn_type, device)

    loss_save_path = f"{input_dir}/model_pth/loss"
    create_dir(loss_save_path)
    for epoch in tqdm(range(500)):
        model_mgcn.train()
        total_loss = 0.0
        optimizer.zero_grad()

        updated_gene_embeddings = model_gene_embedding(gene_similarity_matrix_tensor_dgl, gene_embeddings_tensor)
        for dataset_label in dataset_labels:
            data = features_and_adjacencies_per_dataset[dataset_label]
            image = img_features_per_dataset[dataset_label]
            gene_expression_data = raw_gene_expression_data[dataset_label]

            adj_1hop = data['adj_matrix_1hop_tensor']
            adj_2hop = data['adj_matrix_2hop_tensor']
            adj_3hop = data['adj_matrix_3hop_tensor']

            adj_matrix = convert_adj_matrix_to_dgl_graph(data['cumulative_adj_matrix'], mgcn_gnn_type, device)
            node_features_list = [
                data['filtered_cell_level_features_tensor'],
                data['filtered_patch_level_features_tensor'],
                image
            ]
            node_embeddings = model_mgcn(node_features_list, adj_matrix)

            predicted_expression = torch.matmul(node_embeddings, updated_gene_embeddings.T)
            adj_loss = adjacency_reconstruction_loss(adj_1hop, node_embeddings)
            expression_loss = F.mse_loss(predicted_expression, gene_expression_data)
            sim_loss = node_similarity_loss(adj_1hop, adj_2hop, adj_3hop, node_embeddings)
            dataset_loss = 10 * expression_loss + adj_loss + sim_loss
            total_loss += dataset_loss

        total_loss = total_loss / len(dataset_labels)
        total_loss.backward()
        optimizer.step()
        print("Epoch {} | expression_loss {} | sim_loss {} | adj_loss {} | total_loss {}".format(epoch + 1, expression_loss.item(), sim_loss.item(), adj_loss.item(), total_loss.item()))
        total_loss_values.append(total_loss.item())
        model_save_path = f"{input_dir}/model_pth"
        create_dir(model_save_path)

        if mode == "train":
            if total_loss.item() < min_loss:
                min_loss = total_loss.item()
                save_path = f'{model_save_path}/best_model.pth'
                state = {
                    'model_mgcn': model_mgcn.state_dict(),
                    'model_gene_embedding': model_gene_embedding.state_dict()
                }
                torch.save(state, save_path)


    plot_loss_curve(total_loss_values, label='Total Loss', title='Total Loss', color='blue', file_name=f'{model_save_path}/Total_Loss_Curve.png')

def load_model(input_dims, hidden_dims, output_dim, alpha, input_dir, device, gene_embeddings, mgcn_gnn_type, gene_gnn_type):
    model_gene_embedding = GNN(200, output_dim, gene_gnn_type).to(device)
    model_mgcn = mGCN(input_dims, hidden_dims, output_dim, mgcn_gnn_type, alpha).to(device)
    try:
        checkpoint_path = f'{input_dir}/model_pth/best_model.pth'
        checkpoint = torch.load(checkpoint_path)
        model_mgcn.load_state_dict(checkpoint['model_mgcn'])
        model_gene_embedding.load_state_dict(checkpoint['model_gene_embedding'])
    except FileNotFoundError:
        return None, None
    return model_mgcn, model_gene_embedding

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_random_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    mgcn_gnn_type = "gat"
    gene_gnn_type = "gat"

    dataset_key = "other"
    species = "human"
    input_dir = '/data'
    dataset_labels = ["INT2", "INT3", "INT4", "INT5", "INT6", "INT7", "INT8", "INT9", "INT10", "INT11", "INT12", "INT13", "INT14", "INT15", "INT16", "INT17", "INT18", "INT19"]
    # dataset_labels = ["INT2", "INT3", "INT4", "INT5", "INT6", "INT7", "INT8", "INT9", "INT10"]

    eval_label = "INT1"
    train_dataset_labels = list(set(dataset_labels)- set([eval_label]))
    marker_genes = ''

    create_dir(f"{input_dir}/experiments")
    train_folder = f"{input_dir}/experiments/{eval_label}"
    create_dir(train_folder)

    mode = "train"
    gene_type = "highly_variable"
    n_top_genes = 3000

    gene_embeddings_path = '/workspace/M2TGLGO/biological_database/gene2vec_dim_200_iter_9_w2v.txt'
    gene_embeddings = load_gene_embeddings(gene_embeddings_path)
    gene_embeddings_keys = gene_embeddings.keys()

    features_and_adjacencies_per_dataset, analyzed_genes, spots_used_per_dataset = check_preprocess_data(input_dir, dataset_key, dataset_labels, train_dataset_labels, gene_type, n_top_genes, gene_embeddings_keys, marker_genes, mgcn_gnn_type, mode, device)
    raw_gene_expression_data = check_gene_expression(input_dir, dataset_key, train_dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device)
    create_comp_data(input_dir, dataset_key, dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device,)
    img_features_per_dataset = check_image_feature(input_dir, dataset_key, dataset_labels, train_dataset_labels, spots_used_per_dataset, gene_type, mode, device)

    GODag_path = "/workspace/M2TGLGO/biological_database/go-basic.obo"
    gene_similarity_matrix = check_gene_similarity_matrix(input_dir, species, gene_type, analyzed_genes, GODag_path)

    train(train_folder, train_dataset_labels, features_and_adjacencies_per_dataset, img_features_per_dataset, raw_gene_expression_data, gene_embeddings, gene_similarity_matrix, mgcn_gnn_type, gene_gnn_type, mode, device)


    mode = 'eval'
    features_and_adjacencies_per_dataset, spots_used_per_dataset = \
        preprocess_data(input_dir, dataset_key, [eval_label], [eval_label], gene_type, n_top_genes, gene_embeddings_keys, marker_genes, mgcn_gnn_type, mode, device)
    raw_gene_expression_data, adata_hvgs = get_gene_expression(input_dir, dataset_key, [eval_label], spots_used_per_dataset, gene_type, analyzed_genes, mode, device)
    img_features_per_dataset = check_image_feature(input_dir, dataset_key, [eval_label], [eval_label], spots_used_per_dataset, gene_type, mode, device)

    input_dims = [14, 3, 2048]
    hidden_dims = [256, 512, 1024]
    output_dim = 1024
    alpha = 0.75

    model_mgcn, model_gene_embedding = load_model(input_dims, hidden_dims, output_dim, alpha, train_folder, device, gene_embeddings, mgcn_gnn_type, gene_gnn_type)
    gene_embeddings_tensor = torch.tensor([gene_embeddings[gene] for gene in analyzed_genes], dtype=torch.float32).to(device)
    gene_similarity_matrix_path = f'{input_dir}/{gene_type}_preprocess_output/5.{gene_type}_gene_similarity_matrix.csv'
    gene_similarity_matrix = pd.read_csv(gene_similarity_matrix_path, index_col=0)
    gene_similarity_matrix_tensor = process_gene_similarity_matrix(analyzed_genes, gene_similarity_matrix, device)
    gene_similarity_matrix_tensor_dgl = convert_adj_matrix_to_dgl_graph(gene_similarity_matrix_tensor, gene_gnn_type, device)

    model_mgcn.eval()
    model_gene_embedding.eval()

    with torch.no_grad():

        # ── [DEBUG 1] gene_embeddings_tensor ──────────────────────────────────
        print("\n[DEBUG 1] gene_embeddings_tensor (입력 유전자 임베딩)")
        print(f"  shape : {gene_embeddings_tensor.shape}")
        print(f"  min   : {gene_embeddings_tensor.min().item():.6f}")
        print(f"  max   : {gene_embeddings_tensor.max().item():.6f}")
        print(f"  mean  : {gene_embeddings_tensor.mean().item():.6f}")
        print(f"  std   : {gene_embeddings_tensor.std().item():.6f}")
        constant_gene_emb = (gene_embeddings_tensor.std(dim=1) == 0).sum().item()
        print(f"  상수 행(유전자) 수: {constant_gene_emb}")

        updated_gene_embeddings = model_gene_embedding(gene_similarity_matrix_tensor_dgl, gene_embeddings_tensor)

        # ── [DEBUG 2] updated_gene_embeddings ─────────────────────────────────
        print("\n[DEBUG 2] updated_gene_embeddings (GNN 통과 후 유전자 임베딩)")
        print(f"  shape : {updated_gene_embeddings.shape}")
        print(f"  min   : {updated_gene_embeddings.min().item():.6f}")
        print(f"  max   : {updated_gene_embeddings.max().item():.6f}")
        print(f"  mean  : {updated_gene_embeddings.mean().item():.6f}")
        print(f"  std   : {updated_gene_embeddings.std().item():.6f}")
        constant_updated = (updated_gene_embeddings.std(dim=1) == 0).sum().item()
        print(f"  상수 행(유전자) 수: {constant_updated}")

        data = features_and_adjacencies_per_dataset[eval_label]
        gene_expression_data = raw_gene_expression_data[eval_label]
        adj_matrix = convert_adj_matrix_to_dgl_graph(data['cumulative_adj_matrix'], mgcn_gnn_type, device)

        # ── [DEBUG 3] node input features ─────────────────────────────────────
        cell_feat = data['filtered_cell_level_features_tensor']
        patch_feat = data['filtered_patch_level_features_tensor']
        img_feat = img_features_per_dataset[eval_label]
        print("\n[DEBUG 3] 노드 입력 피처")
        print(f"  cell_level_features  shape={cell_feat.shape}  min={cell_feat.min().item():.4f}  max={cell_feat.max().item():.4f}  std={cell_feat.std().item():.4f}")
        print(f"  patch_level_features shape={patch_feat.shape}  min={patch_feat.min().item():.4f}  max={patch_feat.max().item():.4f}  std={patch_feat.std().item():.4f}")
        print(f"  image_features       shape={img_feat.shape}  min={img_feat.min().item():.4f}  max={img_feat.max().item():.4f}  std={img_feat.std().item():.4f}")

        node_features_list = [cell_feat, patch_feat, img_feat]
        node_embeddings = model_mgcn(node_features_list, adj_matrix)

        # ── [DEBUG 4] node_embeddings ──────────────────────────────────────────
        print("\n[DEBUG 4] node_embeddings (mGCN 출력)")
        print(f"  shape : {node_embeddings.shape}")
        print(f"  min   : {node_embeddings.min().item():.6f}")
        print(f"  max   : {node_embeddings.max().item():.6f}")
        print(f"  mean  : {node_embeddings.mean().item():.6f}")
        print(f"  std   : {node_embeddings.std().item():.6f}")
        constant_nodes = (node_embeddings.std(dim=1) == 0).sum().item()
        print(f"  상수 행(spot) 수: {constant_nodes} / {node_embeddings.shape[0]}")
        # 노드 임베딩이 spot 간에 동일한지 확인 (열 기준 std)
        col_std = node_embeddings.std(dim=0)
        dead_dims = (col_std == 0).sum().item()
        print(f"  dead dimension(열 std=0) 수: {dead_dims} / {node_embeddings.shape[1]}")

        predicted_expression = torch.matmul(node_embeddings, updated_gene_embeddings.T)

        # ── [DEBUG 5] predicted_expression ────────────────────────────────────
        print("\n[DEBUG 5] predicted_expression (예측 유전자 발현)")
        print(f"  shape : {predicted_expression.shape}")
        print(f"  min   : {predicted_expression.min().item():.6f}")
        print(f"  max   : {predicted_expression.max().item():.6f}")
        print(f"  mean  : {predicted_expression.mean().item():.6f}")
        print(f"  std   : {predicted_expression.std().item():.6f}")
        pred_col_std = predicted_expression.std(dim=0)
        constant_genes_pred = (pred_col_std == 0).sum().item()
        print(f"  상수 열(유전자 예측) 수: {constant_genes_pred} / {predicted_expression.shape[1]}")

        # ── [DEBUG 6] gene_expression_data (ground truth) ─────────────────────
        print("\n[DEBUG 6] gene_expression_data (실제 유전자 발현, ground truth)")
        print(f"  shape : {gene_expression_data.shape}")
        print(f"  min   : {gene_expression_data.min().item():.6f}")
        print(f"  max   : {gene_expression_data.max().item():.6f}")
        print(f"  mean  : {gene_expression_data.mean().item():.6f}")
        print(f"  std   : {gene_expression_data.std().item():.6f}")
        true_col_std = gene_expression_data.std(dim=0)
        constant_genes_true = (true_col_std == 0).sum().item()
        print(f"  상수 열(유전자 실제값) 수: {constant_genes_true} / {gene_expression_data.shape[1]}")

        # ── [DEBUG 7] MSE / PCC ────────────────────────────────────────────────
        mse = F.mse_loss(predicted_expression, gene_expression_data).item()
        pred_np = predicted_expression.cpu().numpy()
        true_np = gene_expression_data.cpu().numpy()
        pcc_list = []
        nan_count = 0
        for i in range(pred_np.shape[1]):
            r, _ = pearsonr(pred_np[:, i], true_np[:, i])
            pcc_list.append(r)
            if np.isnan(r):
                nan_count += 1
        mean_gene_pcc = np.nanmean(pcc_list)

        # Spot-level PCC: 각 spot에 대해 모든 유전자의 예측/실제값 상관관계
        spot_pcc_list = []
        spot_nan_count = 0
        for i in range(pred_np.shape[0]):
            r, _ = pearsonr(pred_np[i, :], true_np[i, :])
            spot_pcc_list.append(r)
            if np.isnan(r):
                spot_nan_count += 1
        mean_spot_pcc = np.nanmean(spot_pcc_list)

        print(f"\n[DEBUG 7] 최종 평가 지표")
        print(f"  Gene-level PCC가 nan인 유전자 수: {nan_count} / {pred_np.shape[1]}")
        print(f"  Spot-level PCC가 nan인 spot 수: {spot_nan_count} / {pred_np.shape[0]}")
        print(f"\n[Eval] Dataset: {eval_label}")
        print(f"[Eval] MSE              : {mse:.6f}")
        print(f"[Eval] Gene-level PCC   : {mean_gene_pcc:.6f}  (nan 제외 평균)")
        print(f"[Eval] Spot-level PCC   : {mean_spot_pcc:.6f}  (nan 제외 평균)")
