from torch.utils.data import DataLoader


def get_data(config, Train, param):
    pass

def get_train_test_data(batch_size):
    train_data = []
    test_data = []
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader
