import os

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from common.env_config import config

# Set device to GPU (RTX 5070ti)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model - replace with your model path
# Option 1: From local directory
# model_path = "../../model/dna_bert_2/pretrained_100_400/1/finetune_dna_bert"
# model = BertForSequenceClassification.from_pretrained("../../model/dna_bert_2/pretrained_100_400/1/finetune_dna_bert", trust_remote_code=True)
model = BertForSequenceClassification.from_pretrained(
    "../../model/dna_bert_2/pretrained_100_400/1/finetune_dna_bert",
    # num_labels=2,
    trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained("../../model/dna_bert_2/pretrained_100_400/1/tokenizer", trust_remote_code=True)

dna = ["ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC",
       "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"]
inputs = tokenizer(dna, truncation=False, return_tensors = 'pt').to(device)

x_train = np.load(os.path.join(config.MY_DATA_DIR, "dna_bert_2/100_400/1/dna_bert_2_cls_train_vector.npy"))

# Extract features
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

    # Get the last hidden state
    last_hidden_state = outputs.hidden_states[-1]
    mean_pooling = torch.mean(last_hidden_state, dim=1)

    # Get CLS token embedding (the first token)
    cls_embedding = last_hidden_state[:, 0, :].cpu().numpy()
    print(f"CLS embedding shape: {cls_embedding.shape}")  # Should be (1, hidden_size)

    print(mean_pooling.shape)

    logits = model.classifier(torch.tensor(x_train[0]).to(device))
    # Convert to probabilities using softmax
    probabilities = F.softmax(logits, dim=0)
    print(f"Probabilities: {probabilities}")

    # Get the predicted class (highest probability)
    predicted_class = torch.argmax(logits).item()
    print(f"Predicted class: {predicted_class}")

