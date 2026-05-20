# DeePhafier — Reimplementation Plan

## Paper Info
- **Title**: DeePhafier: a phage lifestyle classifier using a multilayer self-attention neural network combining protein information
- **Authors**: Yan Miao, Zhenyuan Sun, Chen Lin, Haoran Gu, Chenjing Ma, Yingjian Liang, Guohua Wang
- **Corresponding**: ghwang@nefu.edu.cn (Northeast Forestry University)
- **DOI**: 10.1093/bib/bbae377
- **Journal**: Briefings in Bioinformatics, 25(5), August 2024
- **PMC**: PMC11304974
- **Task**: Binary classification phage lifestyle (virulent vs temperate) từ metagenomic sequences

---

## Kiến trúc tổng quan

```
DNA sequence (variable length)
    │
    ▼
[1] Segmentation: chia thành k=10 đoạn × 300bp (m active, k-m zero-padded)
    │
    ▼
[2] Mỗi đoạn 300bp:
    ├─ Codon split (stride=3) → 100 codons
    ├─ Seq2Vec embedding → 100 × 64
    ├─ FragGeneScan (gene detect) → PSI-BLAST vs SWISS-PROT → PSSM (L×20)
    ├─ Normalize PSSM, zero-pad to 64 dims
    └─ Add to codon embedding → input 100 × 64
    │
    ▼
[3] k × BMSNM (parallel, 4-layer local self-attention + maxpool + residual)
    │
    ▼
[4] Concatenate k outputs (giữ thứ tự gốc)
    │
    ▼
[5] Global Self-Attention (single layer, full attention)
    │
    ▼
[6] Fully Connected (20 hidden units) → Softmax → [virulent, temperate]
```

---

## BMSNM (Basic Multi-layer Self-attention Network Model)

### Cấu trúc 4 layers local self-attention

```
Input (100 × 64)
    │
    ▼
Layer 1: window (2w₁+1), stride=1
    │
    ▼ + maxpool(K,V; κ=5, λ=3)
Layer 2: window (2w₂+1), stride=1
    │
    ▼ + residual L1 → Layer 3 input = L1 + L2
Layer 3: window (2w₃+1), stride=1
    │
    ▼ + residual L1+L2 → Layer 4 input = L1 + L2 + L3
Layer 4: window (2w₄+1), stride=ξ
    │
    ▼
Output dim = n/ξ
```

### Self-attention formula
```
L_ji = softmax(αⱼ · qⱼᵢᵀ · K̃_Ψ(i,wⱼ)) · Ṽ_Ψ(i,wⱼ)ᵀ

α = 1/√d  (d = embedding dim = 64 → α ≈ 0.125)
```

### MaxPooling trên K, V
```
K̃ = maxpool(K; κ=5, λ=3)
Ṽ = maxpool(V; κ=5, λ=3)
```

---

## Hyperparameters TỔNG HỢP

### ✅ Đã biết chính xác (paper + supplementary)

| Param | Value | Nguồn |
|-------|-------|-------|
| Segment length | 300 bp | main |
| Codon stride | 3 | S7 |
| Max codons per segment | 100 | S7 |
| Embedding dim (d) | 64 | main |
| α scaling | 1/√d ≈ 0.125 | S1 |
| BMSNM layers | 4 | main |
| Pool kernel κ | 5 | main |
| Pool stride λ | 3 | main |
| **k** (total segments) | **10** | S2 |
| **m** GA (<300bp) | **1** | S2 |
| **m** GB (300-500bp) | **2** | S2 |
| **m** GC (500-1000bp) | **3** | S2 |
| **m** GD (1000-2000bp) | **5** | S2 |
| **m** GE (>2000bp) | **10** | S2 |
| FC hidden units | 20 | S7 |
| Output classes | 2 (softmax) | main |
| Batch size | 64 | main |
| Learning rate | 0.001 | main |
| Epochs | 300 | main |
| Optimizer | Adam (default) | main |
| PSI-BLAST iter | 3 | main |
| PSI-BLAST E-value | 0.001 | main |

### 🔧 Suy luận hợp lý (KHÔNG có trong paper/supp — chọn theo Poolingformer baseline)

Lý do: Paper chỉ ghi ký hiệu w₁-w₄, ξ nhưng KHÔNG công bố giá trị số. Sau khi đọc 573 dòng supplementary và search web/GitHub, không tìm được. Dùng giá trị baseline từ Poolingformer (reference paper [12]) làm điểm khởi đầu, tune sau nếu cần.

| Param | Inferred value | Lý do |
|-------|---------------|-------|
| **w₁** | 3 | Local context nhỏ (3-mer codon) |
| **w₂** | 5 | Tăng dần receptive field |
| **w₃** | 7 | |
| **w₄** | 9 | Largest local window |
| **ξ** | 2 | Output dim = 100/2 = 50 (sau pool 4 lần với λ=3, output = 100/3⁴ ≈ 1.2) |

**Lưu ý quan trọng**: maxpool(λ=3) áp dụng cho K,V mỗi layer làm giảm kích thước nhanh. Cần kiểm tra lại flow tensor khi implement.

### PSSM Normalization (S4)
```
S'_ij = (S_ij - mean(S_j)) / std(S_j)
i ∈ [1,L], j ∈ [1,20]
```

---

## Dataset

### Training/Testing
| Dataset | Virulent | Temperate | Note |
|---------|----------|-----------|------|
| **MD** (McNair) | 77 | 148 | Manually labeled, reliable |
| **SD** (Song) | 1299 | 535 | NCBI RefSeq, software-labeled |
| **Train** | 1353 | 639 | SD + 70% MD |
| **Test** | 23 | 44 | 30% MD |

- **5-fold CV** trên train set, chia theo complete genomes
- Mỗi fold subsample **20,000 sequences** (10k mỗi class)
- Train per fold: 80,000 sequences (4 sets × 20k)
- Validation per fold: 20,000 sequences

### Length Groups (5 nhóm)
| Group | Length | m |
|-------|--------|---|
| GA | 100-300bp | 1 |
| GB | 300-500bp | 2 |
| GC | 500-1000bp | 3 |
| GD | 1000-2000bp | 5 |
| GE | >2000bp | 10 |

### Real Metagenome (external validation)
- CAMI_high
- CAMI Marine
- Human gut metagenome (SRA052203)

---

## Performance Targets

### 5-fold CV (best avg, GE >2000bp)
- Accuracy: **87.54%**
- Recall: 87.82%
- Precision: 87.34%
- F1: 0.8757

### Real metagenome
| Dataset | Accuracy | Recall | Precision | F1 | AUC |
|---------|----------|--------|-----------|-----|-----|
| CAMI_high | 0.7918 | 0.8038 | 0.8210 | 0.8123 | 0.8110 |
| CAMI Marine | 0.7866 | 0.8064 | 0.8103 | 0.8084 | 0.8207 |
| Human gut | 0.7673 | 0.7626 | 0.7332 | 0.7476 | 0.7727 |

### Evaluation Criteria
- Accuracy = (TP+TN)/(TP+FP+TN+FN)
- Recall = TP/(TP+FN)
- Precision = TP/(TP+FP)
- Specificity = TN/(TN+FP)
- F1 = 2·Precision·Recall/(Precision+Recall)
- AUC (ROC curve)
- Friedman + Nemenyi tests for statistical significance

---

## External Tools cần cài đặt
| Tool | Mục đích | Link |
|------|----------|------|
| **FragGeneScan** | Gene prediction từ DNA | https://omics.informatics.indiana.edu/FragGeneScan/ |
| **PSI-BLAST** | Homology search | NCBI BLAST+ suite |
| **SWISS-PROT** | Reference protein DB | UniProt |
| **Seq2Vec** | Codon embedding (Virtifier) | Reference [12,15] |

---

## Data Sources
- MD: https://doi.org/10.1093/gigascience/giab056
- SD: https://doi.org/10.1007/s40484-019-0187-4
- NCBI RefSeq: https://www.ncbi.nlm.nih.gov/refseq/
- CAMI_high: https://data.cami-challenge.org/camiClient.jar
- CAMI Marine: https://data.cami-challenge.org/participate
- Human gut: NCBI SRA052203

---

## NotebookLM Reference
- **Notebook ID**: `1889d9a4-24ff-47c6-90cf-820b93fedd80`
- **URL**: https://notebooklm.google.com/notebook/1889d9a4-24ff-47c6-90cf-820b93fedd80
- **Sources**:
  - Main paper: `71c891d0-791b-4945-86b2-563df982b12e`
  - Supplementary: `fbbb1e7d-fc82-4112-8a2a-102123f893a8`

---

## ✅ Decisions đã chốt

1. **Framework**: PyTorch
2. **Scope**: Full pipeline (data → preprocess → model → train → eval)
3. **Missing hyperparams (w, ξ)**: ★ **Option B — dùng giá trị suy luận** (w₁=3, w₂=5, w₃=7, w₄=9, ξ=2), tune nếu cần
4. **Language**: Python

---

## Implementation Roadmap

### Phase 1: Data Pipeline
- [ ] Download MD + SD phage genomes
- [ ] Genome cleanup, label verification
- [ ] Train/test split (70/30 trên MD, all SD vào train)
- [ ] 5-fold CV split theo complete genomes
- [ ] Subsample 20k sequences per length group per fold
- [ ] Output: FASTA + labels

### Phase 2: Preprocessing
- [ ] Cài FragGeneScan, BLAST+, download SWISS-PROT
- [ ] Pipeline FragGeneScan → identify genes per sequence
- [ ] Pipeline PSI-BLAST → PSSM matrices (cache to disk)
- [ ] Codon split (stride=3) → 100 codons per 300bp
- [ ] Train Seq2Vec embedding model (64-dim) trên train data
- [ ] PSSM normalization (z-score per column) + zero-pad to 64
- [ ] Combine: codon_embed + protein_embed → 100×64 tensor

### Phase 3: Model (PyTorch)
- [ ] `LocalSelfAttention` module: window (2w+1), stride, maxpool K/V
- [ ] `BMSNM` module: 4 layers + residual (L3 = L1+L2, L4 = L1+L2+L3)
- [ ] `GlobalSelfAttention` module: single layer full attention
- [ ] `DeePhafier` model: k parallel BMSNMs + global + FC + softmax
- [ ] Handle zero-padding cho (k-m) inactive segments

### Phase 4: Training
- [ ] DataLoader với batch=64
- [ ] Adam optimizer, lr=0.001
- [ ] Cross-entropy loss
- [ ] Train 300 epochs per length group (5 models cho 5 nhóm)
- [ ] 5-fold CV per length group
- [ ] Model checkpointing

### Phase 5: Evaluation
- [ ] Metrics: Accuracy, Recall, Precision, Specificity, F1, AUC
- [ ] Test trên 30% MD test set
- [ ] Test trên CAMI_high, CAMI Marine, Human gut
- [ ] Compare vs DeePhage, PhagePred
- [ ] Friedman + Nemenyi statistical tests

### Phase 6: Inference
- [ ] CLI tool: input FASTA → predict virulent/temperate
- [ ] Auto-detect length group, route to correct model
- [ ] Output: prediction + confidence score

---

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| w/ξ values khác paper | Dùng baseline B, hyperparameter tune trên val set nếu accuracy < 80% |
| Tensor shape mismatch sau pool | Verify shape sau mỗi layer khi implement |
| PSI-BLAST chậm (millions sequences) | Cache PSSM, parallel processing, có thể dùng MMseqs2 thay nếu cần |
| Seq2Vec không có pretrained | Train from scratch trên train set, hoặc dùng Virtifier embedding |
| Imbalanced classes (1.6k vs 0.6k genomes) | Đã balance ở subsample stage (10k:10k) |
| GPU memory với k=10 BMSNMs | Gradient checkpointing, batch size adjustment |

---

## Next Step

Sẵn sàng implement. Có thể chuyển sang `/spx-ff` để tạo OpenSpec change với tasks chi tiết, hoặc `/spx-apply` để bắt đầu code trực tiếp.
