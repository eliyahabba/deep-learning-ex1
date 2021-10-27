import argparse
import os

from pyhocon import ConfigFactory
import numpy as np
import torch
from torch import optim

from data_iterator import read_data_from_files, get_data_for_training

from stats import Stats
import torch.nn as nn

import importlib

module = importlib.import_module('models')
classes = ['positive', 'negative']


def save_test_eval_to_tensorboard(stats, total_loss, correct, total, epoch, class_total, class_correct):
    stats.summary_writer.add_scalar('test/loss', total_loss, epoch)
    stats.summary_writer.add_scalar('test/acc', 100 * correct / total, epoch)
    stats.summary_writer.add_scalar('test/correct', correct, epoch)
    w_acc = np.average([100 * class_correct[i] / class_total[i] for i in range((len(classes)))])
    stats.summary_writer.add_scalar('test/weight_acc', w_acc, epoch)
    print('Test Accuracy of the model: {} % after %{} epochs'.format(100 * correct / total, epoch))
    print('weight Accuracy of the model: {} % after %{} epochs'.format(w_acc, epoch))

    for i in range(len(classes)):
        stats.summary_writer.add_scalar('test/correct_%s' % classes[i], class_correct[i], epoch)
        stats.summary_writer.add_scalar('test/acc_%s' % classes[i], 100 * class_correct[i] / class_total[i],
                                        epoch)


def calculate_model_results(criterion, test_labels, outputs, total, correct, total_loss, class_total, class_correct):
    _, predicted = torch.max(outputs.detach(), 1)
    test_labels_tensor = torch.as_tensor(test_labels)

    total += test_labels_tensor.size(0)
    correct += (predicted == test_labels_tensor).sum().item()
    total_loss += criterion(outputs, test_labels_tensor)

    c = (predicted == test_labels_tensor).squeeze()
    for i in range(len(test_labels_tensor)):
        label = test_labels_tensor[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
    return total, correct, total_loss, class_correct, class_total


def eval_test_data(net, test_loader, stats, epoch, criterion):
    with torch.no_grad():
        correct, total, total_loss = 0, 0, 0
        class_correct, class_total = [0] * len(classes), [0] * len(classes)

        for test_data in test_loader:
            test_inputs, test_labels = test_data['encoded_gene'], test_data['label']
            outputs = net(test_inputs)
            total, correct, total_loss, class_correct, class_total = calculate_model_results(criterion, test_labels,
                                                                                             outputs, total, correct,
                                                                                             total_loss, class_total,
                                                                                             class_correct)

        # save to tensorboard object
        save_test_eval_to_tensorboard(stats, total_loss, correct, total, epoch, class_total, class_correct)


def train(model_path, config):
    net = getattr(module, config['model'])()

    lr = config['learning_rate']
    epochs = config['epochs']
    batch_size = config['batch_size']

    # choose the optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(params=net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)

    # get the data
    data_text, data_label = read_data_from_files()
    train_loader, test_loader = get_data_for_training(data_text, data_label)
    print('train_loader len is {}'.format(len(train_loader.dataset)))
    print('test_loader len is {}'.format(len(test_loader.dataset)))

    # choose a loss function
    criterion = nn.CrossEntropyLoss()

    stats_keys = ['loss']
    print_step = 1
    stats = Stats(stats_keys, log_dir=model_path, print_step=print_step, prefix='train/')
    step = 0
    for epoch in range(epochs):

        for data in train_loader:
            step += 1
            inputs, inputs_labels = data['encoded_gene'], data['label']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.float())

            loss = criterion(outputs, inputs_labels)
            loss.backward()
            optimizer.step()

            # print statistics of train
            stats.summary_writer.add_scalar('train/loss', loss, step)

            net.eval()
            if step % config['checkpoint_every'] == 0:
                torch.save(net.state_dict(), os.path.join(model_path, '%d.ckpt' % step))

            eval_test_data(net, test_loader, stats, step, criterion)


def get_training_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config args for train model')
    parser.add_argument('--config-file', default=os.path.join(os.getcwd(), 'config.conf'))
    args = parser.parse_args()

    model_path = os.path.join(os.getcwd(), args.config)
    config = ConfigFactory.parse_file(args.config_file)[args.config]
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path, config


if __name__ == '__main__':
    model_path, config = get_training_params()
    train(model_path, config)
