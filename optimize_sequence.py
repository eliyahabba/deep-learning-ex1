# from __future__ import print_function


import argparse

import numpy as np
import torch
import torch.nn as nn

from data_iterator import PROTEIN_ALPHABET
import random

LEN_PEPTIDE = 9


def add_grad_to_exmple(encode_pep, eps):
    encode_pep_grad = encode_pep.grad.data
    new_encode_pep = encode_pep - encode_pep_grad * eps
    new_encode_pep_argmax = torch.argmax(new_encode_pep, dim=1)
    encode_pep.data = torch.zeros(new_encode_pep.shape).scatter(1, new_encode_pep_argmax.unsqueeze(1), 1.0).data


def optimize_sequence(ckpt, pep, eps):
    model = torch.load(ckpt)
    model.eval()
    encode_peptide = [[0. if char != letter else 1. for char in PROTEIN_ALPHABET] for letter in pep]
    encode_peptide = torch.tensor(encode_peptide, requires_grad=True)
    decode_peps = [np.apply_along_axis(lambda x: PROTEIN_ALPHABET[np.argmax(x)], 1, encode_peptide.detach().numpy())]
    predict_detect_count = 0
    NOT_DETECT, DETECT = 0, 1
    iterations_num = 0
    iterations_counter = [iterations_num]
    while predict_detect_count < 10:
        output = model(encode_peptide.unsqueeze(0))
        predict_detect_count = predict_detect_count + 1 if output.squeeze(0)[NOT_DETECT] < output.squeeze(0)[
            DETECT] else 0

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, torch.tensor([1]))

        loss.backward()
        add_grad_to_exmple(encode_peptide, eps)

        decode_pep = np.apply_along_axis(lambda x: PROTEIN_ALPHABET[np.argmax(x)], 1, encode_peptide.detach().numpy())
        iterations_num += 1
        if not np.array_equal(decode_pep, decode_peps[-1]):
            decode_peps.append(decode_pep)
            iterations_counter.append(iterations_num)
            # iterations_num = 0
        # else:
        # iterations_num += 1
    return decode_peps, iterations_counter


def create_random_peptide():
    return ''.join(random.choice(PROTEIN_ALPHABET) for _ in range(LEN_PEPTIDE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', help="The path of the saved ckpt of the model",
                        default="C:\\Users\\t-eliyahabba\\PycharmProjects\\deep_learing\\deep-learning-ex1\\14.ID1_NeuralNetwork_adam_lr_0.0005\\12.epochs_ckpt")
    args = parser.parse_args()
    eps = 0.000001
    epsilons = [1, 0.1,0.01,0.001,0.0001,0.000001]
    epsilons = [0.000001]
    random_peptide = create_random_peptide()
    print(f"We start with random peptide: {random_peptide}")

    for eps in epsilons:
        detected_peptide, iterations_counter = optimize_sequence(args.ckpt, random_peptide, eps)
        for iterations_num, peptide_iter in zip(iterations_counter,
                                                np.apply_along_axis(lambda x: ''.join(x), 1, detected_peptide)[1:]):
            print(f'After {iterations_num + 1} iteration with eps={eps} the string is {peptide_iter}')
        print(f"The model's output on the final peptide {''.join(peptide_iter)} is 'detected'\n")
