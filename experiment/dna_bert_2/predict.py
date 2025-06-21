import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification

from common.env_config import config

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # For binary classification, confusion matrix is [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'f1_score': f1
    }

class EmbeddedDNA(Dataset):
    def __init__(self, sequences, label):
        self.sequences = sequences
        self.label = label

    def __getitem__(self, index):
        return self.sequences[index], self.label[index]

    def __len__(self):
        return len(self.sequences)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        "../../model/dna_bert_2/pretrained_100_400/1/finetune_dna_bert",
        trust_remote_code=True
    ).to(device)

    x_train = np.load(os.path.join(config.MY_DATA_DIR, "dna_bert_2/100_400/1/dna_bert_2_cls_val_vector.npy"))
    y_train = np.load(os.path.join(config.MY_DATA_DIR, "dna_bert_2/100_400/1/y_val.npy"))
    train_ds = EmbeddedDNA(sequences=x_train, label=y_train)
    dataloader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)

    all_predictions = []
    all_labels = []

    with tqdm(total=len(train_ds), desc="Predicting") as pbar:
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.view(-1, 1).to(device)

                outputs = model.classifier(inputs)

                probabilities = F.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(outputs, dim=1)

                all_predictions.extend(predicted_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.update(len(inputs))

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    count = Counter(all_predictions)
    print(count)
    all_labels = np.array(all_labels)

    val_metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))
    print(val_metrics)