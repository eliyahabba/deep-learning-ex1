import argparse
import importlib
import os

import numpy as np
import torch
import torch.nn as nn
from pyhocon import ConfigFactory, HOCONConverter
from sklearn.metrics import accuracy_score, confusion_matrix, \
    f1_score, recall_score, precision_score, balanced_accuracy_score
from torch import optim
import seaborn as sns
import matplotlib.pyplot as plt

from data_iterator import read_data_from_files, get_data_for_training
from stats import Stats

module = importlib.import_module('models')

RANDOM_SEED = 0
classes = ['not detect', 'detect']


def define_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_test_metrics_to_tensorboard(stats, epoch, loss, metrics):
    stats.summary_writer.add_scalar('test/loss', loss, epoch)
    stats.summary_writer.add_scalar('test/acc', metrics['acc'], epoch)
    stats.summary_writer.add_scalar('test/acc_0', metrics['acc_0'], epoch)
    stats.summary_writer.add_scalar('test/acc_1', metrics['acc_1'], epoch)
    stats.summary_writer.add_scalar('test/weight_acc', metrics['weight_acc'], epoch)
    stats.summary_writer.add_scalar('test/recall', metrics['recall'], epoch)
    stats.summary_writer.add_scalar('test/precision', metrics['precision'], epoch)
    stats.summary_writer.add_scalar('test/f1', metrics['f1'], epoch)


def save_test_metrics_to_file(epoch, loss, metrics):
    # print results to txt file
    permission = 'a'
    with open(f"{model_path}/results.txt", permission) as f:
        # print(f"Epoch {epoch}:", file=f)
        print(f"test loss: {loss}", file=f)
        print(metrics, file=f)


def save_train_loss_to_file(epoch, loss):
    # print results to txt file
    permission = 'a'
    if epoch == 1:
        permission = 'w'
    with open(f"{model_path}/results.txt", permission) as f:
        print(f"Epoch {epoch}:", file=f)
        print(f"train loss: {loss}", file=f)


def eval_test_data(net, test_loader, stats, epoch, criterion, show_confusion_matrix=False):
    running_loss = 0.0
    y_true = torch.Tensor()
    y_pred = torch.Tensor()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs, test_labels = test_data['encoded_gene'], test_data['label']
            outputs = net(test_inputs)
            y_true = torch.cat((y_true, test_labels))
            y_pred = torch.cat((y_pred, torch.argmax(outputs.detach(), dim=1)))
            if config['loss'] == 'MSELoss':
                outputs = torch.argmax(outputs.detach(), dim=1).type(torch.DoubleTensor)
            loss = criterion(outputs, test_labels)
            running_loss += loss.item()

        # print avg batch loss
        avg_loss = round(running_loss / (len(test_loader)), 4)
        print(f"test loss: {avg_loss}")

        # get metrics
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        metrics = {'acc': round(accuracy_score(y_true=y_true, y_pred=y_pred), 4),
                   'weight_acc': round(balanced_accuracy_score(y_true=y_true, y_pred=y_pred), 4),
                   'acc_0': round(conf_mat[0][0] / conf_mat[0].sum(), 4),
                   'acc_1': round(conf_mat[1][1] / conf_mat[1].sum(), 4),
                   'recall': round(recall_score(y_true=y_true, y_pred=y_pred), 4),
                   'precision': round(precision_score(y_true=y_true, y_pred=y_pred), 4),
                   'f1': round(f1_score(y_true=y_true, y_pred=y_pred), 4),
                   'confusion_matrix': conf_mat}
        print(f"f1: {metrics['f1']}")
        print(f"acc: {metrics['acc']}")
        if show_confusion_matrix:
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
            plt.title(f'confusion_matrix peptides after {epoch} epochs', fontsize=14)  # title with fontsize 20
            plt.xlabel('Predicted labels', fontsize=15)  # x-axis label with fontsize 15
            plt.ylabel('True labels', fontsize=15)  # y-axis label with fontsize 15
            plt.show()
        print()

        save_test_metrics_to_file(epoch, avg_loss, metrics)
        save_test_metrics_to_tensorboard(stats, epoch, avg_loss, metrics)


def train(model_path, config):
    net = getattr(module, config['model'])()

    lr = config['learning_rate']
    epochs = config['epochs']
    batch_size = config['batch_size']

    # choose the optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=config['momentum'])

    # read the batch_data from the files
    data_text, data_label = read_data_from_files()

    # Preparing the batch_data for training
    train_loader, test_loader = get_data_for_training(data_text, data_label, batch_size)
    print('train_loader len is {}'.format(len(train_loader.dataset)))
    print('test_loader len is {}'.format(len(test_loader.dataset)))
    print("###########################################################")

    # choose a loss function
    if config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise Exception("not valis loss")

    # TODO change to batch loss
    stats_keys = ['loss']
    print_step = 1
    stats = Stats(stats_keys, log_dir=model_path, print_step=print_step, prefix='train/')
    step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_data in train_loader:
            # batch_data dim = batch_size * 9 * 26
            step += 1
            inputs, inputs_labels = batch_data['encoded_gene'], batch_data['label']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            if config['loss'] == 'MSELoss':
                outputs = torch.argmax(outputs.detach(), dim=1).type(torch.DoubleTensor)
                outputs.requires_grad = True
                inputs_labels = inputs_labels.type(torch.DoubleTensor)
            loss = criterion(outputs, inputs_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print statistics of train
            stats.summary_writer.add_scalar('train/batch_loss', loss, step)

        avg_epoch_loss = round(running_loss / len(train_loader), 4)
        stats.summary_writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, step)
        print(f"Epoch {epoch + 1}:")
        print(f"train loss: {avg_epoch_loss}")
        save_train_loss_to_file(epoch + 1, avg_epoch_loss)

        net.eval()
        torch.save(net, os.path.join(model_path, '%d.epochs_ckpt' % epoch))
        eval_test_data(net, test_loader, stats, epoch=epoch + 1, criterion=criterion)


def get_training_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config args for train model')
    parser.add_argument('--config-file', default=os.path.join(os.getcwd(), 'config.conf'))
    args = parser.parse_args()

    model_path = os.path.join(os.getcwd(), args.config)
    config = ConfigFactory.parse_file(args.config_file)[args.config]
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # save model config file
    with open(f"{model_path}/{args.config}_config.conf", "w") as f:
        f.write(f'{args.config}=')
        f.write(HOCONConverter.to_json(config))

    return model_path, config


if __name__ == '__main__':
    # define_random_seed(0)
    model_path, config = get_training_params()
    train(model_path, config)
