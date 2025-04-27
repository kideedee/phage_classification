# Pipeline xử lý dữ liệu
1. Tổng hợp dữ liệu DNA thực khuẩn (metadata)
2. Gộp dữ liệu meta data của bộ dữ liệu vào một file
3. Chia bộ dữ liệu thành 2 tập train, valid
4. Tải dữ liệu từ NCBI dạng genbank thông qua accession number được lưu trong file metadata train, valid
5. Gộp các chuỗi DNA vào file fasta theo từng tập: train.fasta, valid.fasta
6. Tiền xử lý dữ liệu
7. Huấn luyện
8. Đánh giá

# Các phương pháp biểu diễn dữ liệu
1. One-hot encoding


# PhageAI
1. Train-validation-test split
2. Reverse complement augmentation
3. Efficient DNA word embedding
4. Classification with ML
    - Efficient feature selection
    - Supervised learning
5. Model evaluation

## Train-validation-test split
- Stratified cross validation with k=10
- 

## Bộ dữ liệu
- 278 virulent và 174 temperate
- fasta format

## Efficient DNA word embedding
- Apply k-mer structure with sliding window approach using constant k=6
- Using Word2Vec with the Skip-gram model, vocabulary size V=4096, 300 fixed-size numeric vector space
- Train 20 epochs and optimize with negative sampling instead of hierarchy softmax function
- Train Word2Vec with config: {size=300, window=5, min_count=1, sample=1e-3, sg=1, hs=0, epochs=20, negative=5, word_ngrams=1, random_state=42}  

## Efficient feature selection
- Use Feature ranking with recursive feature elimination and cross-validated selection of the best number of features using SVM estimator to chosen 150 nominal features from 300

## Supervised learning
- Use Bayesian optimization to tune hyperparameters