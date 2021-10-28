import argparse

import numpy as np
import torch

from data_iterator import PeptidesDataset

GEN_SIZE = 9
K = 5


def get_spike_protein():
    file = open('spike_data.txt', mode='r')
    proteins = file.read()
    return "".join(proteins.split())


def eval_model_on_evert_seqs(model, proteins_data, k=K):
    all_sequences = [proteins_data[i:i + GEN_SIZE] for i in range(len(proteins_data) - GEN_SIZE + 1)]
    one_hot_all_sequences = [PeptidesDataset.one_hot_encoder(sequence) for sequence in all_sequences]
    outputs = model(torch.stack(one_hot_all_sequences, dim=0))

    _, predicted = torch.max(outputs.detach(), 1)
    top_k_detected = torch.topk(outputs[:, 1], k)
    top_k_indices = top_k_detected[1].numpy()
    top_k_aminos = np.array(all_sequences)[top_k_indices]
    return top_k_aminos


def evaluate(ckpt):
    proteins_data = get_spike_protein()
    model = torch.load(ckpt)
    model.eval()
    return eval_model_on_evert_seqs(model, proteins_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help="The path of the saved ckpt of the model",
                        default="C:\\Users\\t-eliyahabba\\PycharmProjects\\deep_learing\\deep-learning-ex1\\15.ID1_NeuralNetwork_adam_lr_0.00001\\5.epochs_ckpt")
    args = parser.parse_args()
    top_k_aminos = evaluate(args.ckpt)
    print(f"The top K={K} detectable peptides are:", *top_k_aminos, sep="\n")
