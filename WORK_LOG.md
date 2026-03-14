# M2TGLGO 프로젝트 작업 로그

## 프로젝트 개요
Spatial transcriptomics 유전자 발현 예측 모델(mGCN + GAT 기반).
H&E 조직 이미지에서 유전자 발현값을 예측한다.

- **모델**: mGCN (multi-Graph Convolutional Network) with GAT
- **입력**: cell-level features (HoverNet), patch-level features, image features (ResNet50)
- **출력**: 유전자 발현값 예측
- **데이터**: HEST-1k benchmark (spatial transcriptomics + H&E paired data)

---

## 환경 정보

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA L40 (1개, index 0) |
| Container memory | 40GB (cgroup limit) |
| Shared memory | 64MB (/dev/shm) |
| Python | 3.10 |
| CUDA | available |
| Data directory | `/data/` |
| Project directory | `/workspace/M2TGLGO/` |
| Benchmark data | `/research/05-WSI/Rawdata/hest1k/hest_bench/` |

---

## 데이터 구조

### 암종별 데이터셋 (hest_bench)
| 암종 | 디렉토리 | 슬라이드 | 유전자 수 | 플랫폼 |
|------|----------|---------|----------|--------|
| CCRCC | `/hest_bench/CCRCC/` | INT1~INT24 | 17,943~36,601 | Whole transcriptome |
| HCC | `/hest_bench/HCC/` | NCBI642, NCBI643 | 36,601 | Whole transcriptome |
| LYMPH_IDC | `/hest_bench/LYMPH_IDC/` | NCBI681~NCBI684 | 33,931 | Whole transcriptome |
| PRAD | `/hest_bench/PRAD/` | 23 slides | 33,538 | Whole transcriptome |
| READ | `/hest_bench/READ/` | 4 slides | 36,601 | Whole transcriptome |
| COAD | `/hest_bench/COAD/` | 4 slides | ~541 | Targeted panel |
| IDC | `/hest_bench/IDC/` | 4 slides | ~541 | Targeted panel |
| LUNG | `/hest_bench/LUNG/` | 2 slides | ~541 | Targeted panel |
| PAAD | `/hest_bench/PAAD/` | 3 slides | 538~541 | Targeted panel |
| SKCM | `/hest_bench/SKCM/` | 2 slides | ~541 | Targeted panel |

### /data/ 디렉토리 구조
```
/data/
├── INT1~INT24/          # CCRCC slides
│   ├── patches/         # H&E image patches (256x256 png)
│   ├── segment/         # HoverNet segmentation output
│   │   ├── mat/         # instance segmentation maps (.mat)
│   │   ├── json/        # cell info with type, centroid, contour
│   │   ├── overlay/     # visualization
│   │   └── qupath/      # QuPath format
│   ├── feature/         # extracted cell/patch features
│   │   ├── cell_level_features.csv
│   │   ├── patch_level_features.csv
│   │   └── cell_count.csv
│   └── ...
├── NCBI642~NCBI684/     # HCC, LYMPH_IDC slides (same structure)
├── highly_variable_preprocess_output/   # 캐시 디렉토리
│   ├── 1.features_and_adjacencies_per_dataset.pkl
│   ├── 2.img_features_per_dataset.pkl
│   ├── 3.spots_used_per_dataset.pkl
│   ├── 4.raw_highly_variable_gene_expression_data.pkl
│   ├── 4.shared_highly_variable_genes.npy
│   ├── 4.shared_highly_variable_genes.pkl
│   ├── 4.shared_highly_variable_genes.txt
│   ├── 5.highly_variable_gene_similarity_matrix.csv
│   └── 5.highly_variable_gene_similarity_results.pkl
└── model_pth/           # 학습된 모델 저장
```

---

## 주요 파일 설명

### 스크립트
| 파일 | 설명 |
|------|------|
| `scripts/train_and_eval_debug.py` | 메인 학습/평가 스크립트 (디버그 출력 포함) |
| `scripts/train_and_eval.py` | 원본 학습/평가 스크립트 |
| `scripts/utils.py` | 원본 유틸리티 (seurat_v3 flavor HVG) |
| `scripts/utils_seurat.py` | **수정본** 유틸리티 (seurat flavor HVG) |
| `scripts/calculate_gene_similarity.py` | GO term 기반 유전자 유사도 계산 |
| `scripts/mgcn.py` | mGCN, GNN 모델 정의 |
| `scripts/gene2vec.py` | Gene2Vec 임베딩 로더 |
| `scripts/find_constant_genes.py` | NCBI683 상수 유전자 탐색 스크립트 |
| `preprocess/patch_cell_featere_extract.py` | HoverNet mask → cell feature 추출 |

### 생물학적 데이터베이스
| 파일 | 설명 |
|------|------|
| `biological_database/gene2vec_dim_200_iter_9_w2v.txt` | Gene2Vec 임베딩 (200차원) |
| `biological_database/go-basic.obo` | GO DAG (2024-06-17) |

### HoverNet
| 파일 | 설명 |
|------|------|
| `scripts/hover_net/` | hover_net GitHub repo clone |
| `scripts/hover_net/run_hovernet.sh` | HoverNet 실행 스크립트 |
| `scripts/hover_net/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar` | PanNuke pretrained model (151MB) |
| `scripts/hover_net/type_info.json` | 6개 세포 타입 정의 |

---

## utils.py → utils_seurat.py 변경 사항

`utils_seurat.py`는 `utils.py`의 복사본으로 다음이 변경됨:

### 1. HVG 선택 flavor 변경 (line ~374-376)
```python
# 변경 전 (utils.py):
sc.pp.highly_variable_genes(adata_integrated, flavor="seurat_v3", n_top_genes=n_top_genes)

# 변경 후 (utils_seurat.py):
sc.pp.normalize_total(adata_integrated, target_sum=1e4)
sc.pp.log1p(adata_integrated)
sc.pp.highly_variable_genes(adata_integrated, flavor="seurat", n_top_genes=n_top_genes)
```
- **이유**: `seurat_v3`는 raw count 데이터에서 dense matrix 변환 필요 → 18개 데이터셋 합치면 OOM (40GB 제한)
- **해결**: `seurat` flavor은 log-normalized 데이터 기반 → 메모리 효율적

### 2. GO term fetch worker 수 변경 (line ~627)
```python
# 변경 전: num_workers_fetch=80
# 변경 후: num_workers_fetch=1
```
- **이유**: multiprocessing worker에서 mygene.info API SSL 연결 실패
- **해결**: 순차 처리 (num_workers=1)로 변경

---

## calculate_gene_similarity.py 변경 사항

### 1. prefetch_go_terms 순차 모드 추가 (line ~85-97)
```python
def prefetch_go_terms(ensembl_ids, num_workers=8):
    go_terms_mapping = {}
    if num_workers <= 1:
        # Sequential mode: avoids SSL issues in forked processes
        for ensembl_id in tqdm(ensembl_ids, desc="Fetching GO terms"):
            _, go_terms = fetch_go_terms(ensembl_id)
            go_terms_mapping[ensembl_id] = go_terms
    else:
        with mp.Pool(num_workers) as pool:
            ...
```
- **이유**: forked process에서 httpx SSL 연결 실패 (`httpx.ReadError: [SSL: WRONG_VERSION_NUMBER]`)
- **해결**: `num_workers=1`이면 순차 처리

### 2. sleep 시간 변경 (line ~51)
```python
# 변경 전: time.sleep(10)
# 변경 후: time.sleep(1)
```
- **이유**: 2899개 Ensembl ID × 10초 = 8시간 → 1초로 줄이면 ~1시간

---

## train_and_eval_debug.py 현재 설정

```python
from utils_seurat import *          # seurat flavor 사용

dataset_labels = ["INT2"~"INT19"]   # 18개 훈련 데이터셋
eval_label = "INT1"                 # 평가 데이터셋
n_top_genes = 3000                  # HVG 후보 수
gene_type = "highly_variable"
mgcn_gnn_type = "gat"
gene_gnn_type = "gat"

# 학습 하이퍼파라미터
lr = 0.0001
weight_decay = 1e-3
optimizer = AdamW
epochs = 500
hidden_dims = [256, 512, 1024]
output_dim = 1024
alpha = 0.75
```

### 디버그 출력 (eval 시)
- DEBUG 1: 입력 유전자 임베딩 통계
- DEBUG 2: GNN 통과 후 유전자 임베딩 통계
- DEBUG 3: 노드 입력 피처 (cell, patch, image) 통계
- DEBUG 4: mGCN 출력 노드 임베딩 통계
- DEBUG 5: 예측 유전자 발현 통계
- DEBUG 6: 실제 유전자 발현 (ground truth) 통계
- DEBUG 7: MSE, Gene-level PCC, Spot-level PCC

---

## 실험 결과

### 실험 1: INT2-INT10 (9개) → INT1, seurat_v3, n_top_genes=50
- HVG: 21개
- MSE: 1.487443
- PCC: 0.041951

### 실험 2: INT2-INT19 (18개) → INT1, seurat, n_top_genes=50
- HVG: 37개
- MSE: 0.173322 / Gene PCC: 0.023053 / Spot PCC: N/A
- 상수 유전자: 6/37 (ground truth)

### 실험 3: INT2-INT19 (18개) → INT1, seurat, n_top_genes=3000, 100 epoch
- HVG: 2,457개
- MSE: 0.217720 / Gene PCC: 0.020849 / Spot PCC: 0.370155
- 상수 유전자: 575/2,457 (23.4%)

### 실험 4: INT2-INT19 (18개) → INT1, seurat, n_top_genes=3000, 500 epoch
- HVG: 2,457개
- **MSE: 0.200045 / Gene PCC: 0.017090 / Spot PCC: 0.456271**
- 상수 유전자: 575/2,457

| 설정 | MSE | Gene PCC | Spot PCC |
|------|-----|----------|----------|
| 9 datasets, 21 HVG, 100ep | 1.487 | 0.042 | - |
| 18 datasets, 37 HVG, 100ep | 0.173 | 0.023 | - |
| 18 datasets, 2457 HVG, 100ep | 0.218 | 0.021 | 0.370 |
| **18 datasets, 2457 HVG, 500ep** | **0.200** | **0.017** | **0.456** |

---

## 해결한 문제들

### 1. OOM (Out of Memory) - exit code 137
- **원인**: `seurat_v3` HVG 선택 시 `sc.concat` + dense matrix 변환 → 18개 데이터셋 합치면 40GB 초과
- **해결**: `seurat` flavor으로 변경 (log-normalized 데이터 기반, dense 변환 불필요)

### 2. mygene.info API SSL 오류
- **원인**: multiprocessing forked process에서 httpx SSL 연결 실패
- **해결**: `prefetch_go_terms`에 순차 모드 추가 (`num_workers=1`), sleep 10초→1초

### 3. PCC NaN
- **원인**: ground truth에서 std=0인 유전자 (상수 유전자) → PCC 분모가 0
- **설명**: 훈련 데이터 전체에서 HVG를 선택하므로, eval 데이터에서는 발현이 없을 수 있음
- **현재 처리**: `np.nanmean`으로 NaN 제외 평균

### 4. KeyError in cache
- **원인**: dataset_labels 변경 후 이전 캐시가 남아있음
- **해결**: `/data/highly_variable_preprocess_output/` 캐시 삭제 후 재실행

### 5. Categorical categories must be unique
- **원인**: dataset_labels에 중복값 존재 (INT15가 2번)
- **해결**: 중복 제거

### 6. HoverNet shared memory 부족
- **원인**: /dev/shm이 64MB로 제한됨
- **해결**: `nr_inference_workers=0`, `nr_post_proc_workers=4`, `batch_size=32`로 축소

### 7. HoverNet state_dict 불일치
- **원인**: `nr_types=0`인데 checkpoint에 tp 브랜치 키 존재 → `strict=True` 실패
- **해결**: `infer/base.py`에서 `strict=False`로 변경

---

## 캐시 관리

전처리 결과는 `/data/highly_variable_preprocess_output/`에 캐시됨.
**dataset_labels, n_top_genes, HVG flavor 등을 변경하면 반드시 캐시를 삭제해야 함:**

```bash
rm -f /data/highly_variable_preprocess_output/[1-5]*
```

GO term similarity만 재계산하려면:
```bash
rm -f /data/highly_variable_preprocess_output/5.*
```

---

## Cell Feature 분석 결과

### INT1 cell_level_features.csv 분석
- 전체 spot: 1,084개, feature가 있는 spot: 268개만
- Cell count 중앙값: 0 (75%가 0) — HoverNet이 대부분 spot에서 세포 미검출
- Neighbor features (3개): 95.1%가 0
- Cell-level features (14개): Area, Perimeter 등 형태학적 피처, 합리적 범위
- `patch_cell_featere_extract.py`는 HoverNet의 **모든 세포 타입**을 사용 (type 필터링 없음)
- HoverNet JSON에 `type` 정보가 있음 (1=Neoplastic, 2=Inflammatory, 3=Connective, 4=Dead/Necrosis, 5=Non-neoplastic Epithelial)

### HoverNet 재실행
- `scripts/hover_net/run_hovernet.sh`로 전체 데이터셋에 대해 HoverNet 핵 검출 재실행 가능
- 현재 `nr_types=0` (핵 검출만, 타입 분류 없음)
- pretrained model: PanNuke fast mode (`1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR`)

---

## 유전자 겹침 분석 결과

### 암종 내 슬라이드 간
- Whole transcriptome 플랫폼 (HCC, LYMPH_IDC, PRAD, READ): 100% 동일 유전자 세트
- CCRCC: 49% 교집합 (슬라이드 간 유전자 수 차이 있음)
- Targeted panel: 17~86% (플랫폼/패널 차이)

### 암종 간
- Whole transcriptome끼리: Jaccard 0.82~1.0
- Targeted panel끼리: Jaccard 0.13~0.27
- 전체 교집합: **47개 유전자** (0.1%)

---

## 재현 방법

### 학습 + 평가 실행
```bash
cd /workspace/M2TGLGO/scripts
python train_and_eval_debug.py
```
- 캐시가 있으면 전처리 스킵, 학습+평가만 수행
- 캐시가 없으면 전처리부터 시작 (GO term fetch ~1시간 소요)

### HoverNet 실행
```bash
cd /workspace/M2TGLGO/scripts/hover_net
bash run_hovernet.sh
```

### Cell feature 추출
```bash
cd /workspace/M2TGLGO
python preprocess/patch_cell_featere_extract.py
```
- `main(input_dir='/data', segment_tool='hovernet')` 호출 필요

---

## 상수 유전자 파일
- `/workspace/M2TGLGO/constant_genes_NCBI683.txt`: NCBI683에서 std=0인 313개 유전자 이름
- `scripts/find_constant_genes.py`로 생성
