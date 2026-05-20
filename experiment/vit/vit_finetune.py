import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTConfig, ViTForImageClassification


class CustomDataset(Dataset):
    def __init__(self, h5_path, vectors_key='vectors', labels_key='labels', transform=None):
        self.h5_path = h5_path
        self.vectors_key = vectors_key
        self.labels_key = labels_key
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[labels_key])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            vector = f[self.vectors_key][idx].astype(np.float32)
            label = f[self.labels_key][idx].astype(np.int64)

            # Convert to tensor
            vector = torch.from_numpy(vector)

            # Reshape vector (assuming 2D input)
            if len(vector.shape) == 2:
                reshape_vector = vector.view(1, vector.shape[0], vector.shape[1])
            else:
                reshape_vector = vector

            if self.transform:
                reshape_vector = self.transform(reshape_vector)

            return reshape_vector, label

config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification(config)
original_patch_embed = model.vit.embeddings.patch_embeddings.projection
new_patch_embed = torch.nn.Conv2d(
    in_channels=5,
    out_channels=original_patch_embed.out_channels,
    kernel_size=original_patch_embed.kernel_size,
    stride=original_patch_embed.stride
)

# Initialize with pretrained weights for first 3 channels
with torch.no_grad():
    new_patch_embed.weight[:, :3] = original_patch_embed.weight
    new_patch_embed.weight[:, 3:] = original_patch_embed.weight[:, :2]  # duplicate some channels

model.vit.embeddings.patch_embeddings.projection = new_patch_embed