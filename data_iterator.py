import string
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

BATCH_SIZE = 64


def read_data_from_files():
    with open('neg_A0201.txt', 'r') as f:
        neg_text = f.read().splitlines()
        neg_label = [0] * len(neg_text)
        print(f"number of neg labels: {len(neg_label)}")

    with open('pos_A0201.txt', 'r') as f:
        pos_text = f.read().splitlines()
        pos_label = [1] * len(pos_text)
        print(f"number of pos labels: {len(pos_label)}")

    data_text = neg_text + pos_text
    data_label = neg_label + pos_label
    return data_text, data_label


def get_data_for_training(data_text, data_label, batch_size=BATCH_SIZE):
    # train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data_text, data_label, test_size=0.10, random_state=42)

    # make Dataset
    train_dataset = PeptidesDataset(x_train, y_train)
    test_dataset = PeptidesDataset(x_test, y_test)

    # make DataLoader
    sampler_train = get_sampler(y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler_train)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    return train_loader, test_dataloader


def get_sampler(y_train):
    # count pos and neg
    counter = Counter(y_train)

    # Oversample minority class
    class_sample_count = torch.Tensor([counter[0], counter[1]])
    weights = 1. / class_sample_count.float()
    samples_weights = torch.tensor([weights[t] for t in y_train])

    # check if replacement???
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights))
    return sampler


class PeptidesDataset(Dataset):
    def __init__(self, data_text, data_labels=None):
        self.data_text = data_text
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data_text)

    def __getitem__(self, index: int):
        return dict(gene=self.data_text[index],
                    encoded_gene=self.one_hot_encoder(self.data_text[index]),
                    label=torch.tensor(self.data_labels[index]))

    @staticmethod
    def one_hot_encoder(text):
        alphabet = string.ascii_uppercase
        encoding = torch.Tensor([[0 if char != letter else 1 for char in alphabet] for letter in text])
        return encoding


