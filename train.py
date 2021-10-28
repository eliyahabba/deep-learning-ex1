import argparse
import os

from pyhocon import ConfigFactory, HOCONConverter
import numpy as np
import torch
from torch import optim

from data_iterator import read_data_from_files, get_data_for_training

from stats import Stats
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    f1_score, recall_score, precision_score, balanced_accuracy_score

import importlib

module = importlib.import_module('models')
classes = ['positive', 'negative']

# determine seed
# RANDOM_SEED = 0
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)


# def save_test_eval_to_tensorboard(stats, total_loss, correct, total, epoch, class_total, class_correct):
#     stats.summary_writer.add_scalar('test/loss', total_loss, epoch)
#     stats.summary_writer.add_scalar('test/acc', 100 * correct / total, epoch)
#     stats.summary_writer.add_scalar('test/correct', correct, epoch)
#     w_acc = np.average([100 * class_correct[i] / class_total[i] for i in range((len(classes)))])
#     stats.summary_writer.add_scalar('test/weight_acc', w_acc, epoch)
#     print('Test Accuracy of the model: {} % after {} epochs'.format(round(100 * correct / total, 2), epoch))
#     print('weight Accuracy of the model: {} % after {} epochs'.format(round(w_acc, 2), epoch))
#     print()
#
#     # to tensor board
#     for i in range(len(classes)):
#         stats.summary_writer.add_scalar('test/correct_%s' % classes[i], class_correct[i], epoch)
#         stats.summary_writer.add_scalar('test/acc_%s' % classes[i], 100 * class_correct[i] / class_total[i],
#                                         epoch)

def save_test_metrics_to_tensorboard(stats, epoch, loss, metrics):
    stats.summary_writer.add_scalar('test/loss', loss, epoch)
    stats.summary_writer.add_scalar('test/acc', metrics['acc'], epoch)
    stats.summary_writer.add_scalar('test/acc_0', metrics['acc_0'], epoch)
    stats.summary_writer.add_scalar('test/acc_0', metrics['acc_1'], epoch)
    stats.summary_writer.add_scalar('test/weight_acc', metrics['weight_acc'], epoch)
    stats.summary_writer.add_scalar('test/recall', metrics['recall'], epoch)
    stats.summary_writer.add_scalar('test/precision', metrics['precision'], epoch)
    stats.summary_writer.add_scalar('test/f1', metrics['f1'], epoch)


# def calculate_model_results(criterion, test_labels, outputs, total, correct, total_loss, class_total, class_correct):
#     _, predicted = torch.max(outputs.detach(), 1)
#     test_labels_tensor = torch.as_tensor(test_labels)
#
#     total += test_labels_tensor.size(0)
#     correct += (predicted == test_labels_tensor).sum().item()
#     total_loss += criterion(outputs, test_labels_tensor)
#
#     c = (predicted == test_labels_tensor).squeeze()
#     for i in range(len(test_labels_tensor)):
#         label = test_labels_tensor[i]
#         class_correct[label] += c[i].item()
#         class_total[label] += 1
#     return total, correct, total_loss, class_correct, class_total


def eval_test_data(net, test_loader, stats, epoch, criterion):
    running_loss = 0.0
    y_true = torch.Tensor()
    y_pred = torch.Tensor()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs, test_labels = test_data['encoded_gene'], test_data['label']
            outputs = net(test_inputs)
            loss = criterion(outputs, test_labels)
            running_loss += loss.item()
            y_true = torch.cat((y_true, test_labels))
            y_pred = torch.cat((y_pred, torch.argmax(outputs.detach(), dim=1)))

        # print avg batch loss
        avg_loss = round(running_loss / (len(test_loader) / config['batch_size']), 4)
        print(f"Epoch {epoch}:")
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
        print(metrics)
        print()

        # print results to txt file
        permission = 'a'
        if epoch == 1:
            permission = 'w'
        with open(f"{model_path}/test_results.txt", permission) as f:
            print(f"Epoch {epoch}:", file=f)
            print(f"test loss: {avg_loss}", file=f)
            print(metrics, file=f)

        # save to tensorboard object
        save_test_metrics_to_tensorboard(stats, epoch, avg_loss, metrics)


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
    train_loader, test_loader = get_data_for_training(data_text, data_label, batch_size=batch_size)
    print('train_loader len is {}'.format(len(train_loader.dataset)))
    print('test_loader len is {}'.format(len(test_loader.dataset)))
    print("###########################################################")

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
    model_path, config = get_training_params()
    train(model_path, config)
