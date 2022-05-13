import os
import torch
import numpy as np

from torch.utils.data import Dataset, TensorDataset


class Mydataset(Dataset):
    def __init__(self, img_path, label_path, is_train=True, transform=None):
        self.path = img_path
        self.label_path = label_path
        self.transform = transform
        self.is_train = is_train
        if is_train:
            self.img = os.listdir(self.path)[:50000]
            self.labels = open(self.label_path, 'r').read().split('\n')[:50000]
        else:
            self.img = os.listdir(self.path)[:10000]
            self.labels = open(self.label_path, 'r').read().split('\n')[:10000]

    def __getitem__(self, idx):
        img = np.load(f'{self.path}/{idx}.npy')  # (T, H, W)
        label = torch.tensor(np.array(self.labels[idx].split(), dtype=int))
        img_out = torch.zeros(img.shape)
        if self.transform is not None:
            for i in range(img.shape[0]):
                img_seg = img[i, :, :]  # (H, W)
                img_transform = self.transform(img_seg.astype(np.uint8))  # (H, W) -> (1, H, W)
                img_out[i, :, :] = img_transform.squeeze(0)  # (1, H, W) -> (H, W)
        else:
            img_out = torch.tensor(img)
        return img_out, label

    def __len__(self):
        return len(self.img)


def Mytensordataset(img_path, label_path, is_train=True, transform=None):

    tensor_img = torch.load(img_path)
    tensor_label = torch.load(label_path)
    if transform is not None:
        H = tensor_img.size(-2)
        W = tensor_img.size(-1)
        T = tensor_img.size(1)
        tensor_img = tensor_img.view(-1, H, W).unsqueeze(1)  # (NxT, 1(C), H, W)
        for i in range(tensor_img.size(0)):
            tensor_img[i, 0, :, :] = transform(tensor_img[i, 0, :, :].data.numpy().astype(np.uint8)).squeeze(0)
        tensor_img = tensor_img.squeeze(1).view(-1, T, H, W).type(torch.float)

    if is_train:
        return TensorDataset(tensor_img, tensor_label)  # (N, T, H, W), (N, T)
    else:
        return TensorDataset(tensor_img, tensor_label)  # (N, T, H, W), (N, T)

def collate_fn(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.cat([item[1] for item in batch], dim=0)
    img = img.unsqueeze(1)  # (BxT, C=1, H, W)
    return img, label