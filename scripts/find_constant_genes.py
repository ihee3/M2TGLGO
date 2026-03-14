"""
NCBI683 eval dataset에서 모든 spot에 걸쳐 발현값이 동일한(constant) 유전자를 찾아 저장하는 스크립트.
train_and_eval_debug.py의 eval 전처리 파이프라인을 재현한다.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pickle

from utils import preprocess_data, get_gene_expression
from gene2vec import load_gene_embeddings

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

input_dir = '/data'
dataset_key = "other"
gene_type = "highly_variable"
n_top_genes = 3000
mgcn_gnn_type = "gat"
eval_label = "NCBI683"

gene_embeddings_path = '/workspace/M2TGLGO/biological_database/gene2vec_dim_200_iter_9_w2v.txt'
gene_embeddings = load_gene_embeddings(gene_embeddings_path)
gene_embeddings_keys = gene_embeddings.keys()

# analyzed_genes는 shared_highly_variable_genes.npy에 저장되어 있음
analyzed_genes_path = f'{input_dir}/{gene_type}_preprocess_output/4.shared_{gene_type}_genes.npy'
analyzed_genes = np.load(analyzed_genes_path, allow_pickle=True).tolist()
print(f"analyzed_genes loaded: {len(analyzed_genes)} genes")

marker_genes = ''

# eval 전처리: NCBI683에 대해 preprocess_data 실행
print("\nRunning preprocess_data for NCBI683 (eval mode)...")
features_and_adjacencies_per_dataset, spots_used_per_dataset = preprocess_data(
    input_dir, dataset_key, [eval_label], [eval_label],
    gene_type, n_top_genes, gene_embeddings_keys,
    marker_genes, mgcn_gnn_type, 'eval', device
)
print(f"spots_used_per_dataset keys: {list(spots_used_per_dataset.keys())}")
print(f"Number of spots for {eval_label}: {len(spots_used_per_dataset[eval_label])}")

# 유전자 발현 데이터 로드
print("\nRunning get_gene_expression for NCBI683...")
raw_gene_expression_data, adata_hvgs = get_gene_expression(
    input_dir, dataset_key, [eval_label],
    spots_used_per_dataset, gene_type, analyzed_genes, 'eval', device
)

gene_expr_tensor = raw_gene_expression_data[eval_label]
print(f"gene_expression_data shape: {gene_expr_tensor.shape}")

# std==0인 유전자(constant 유전자) 찾기
col_std = gene_expr_tensor.std(dim=0)
constant_mask = (col_std == 0)
n_constant = constant_mask.sum().item()
print(f"\nConstant genes (std==0 across all spots): {n_constant} / {gene_expr_tensor.shape[1]}")

# 유전자 이름 추출
constant_gene_indices = torch.where(constant_mask)[0].tolist()
constant_gene_names = [analyzed_genes[i] for i in constant_gene_indices]

# 저장
output_path = '/workspace/M2TGLGO/constant_genes_NCBI683.txt'
with open(output_path, 'w') as f:
    for gene in constant_gene_names:
        f.write(gene + '\n')

print(f"\nSaved {len(constant_gene_names)} constant gene names to: {output_path}")
print("First 10 genes:", constant_gene_names[:10])
