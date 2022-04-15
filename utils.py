import os
import numpy as np
import random
import warnings
import argparse
import torch
import pickle
import datetime
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Set Seed " + str(seed) + " Done !!!")


def str_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def data_reader(dataset_name):
    train_filename = "datasets/" + dataset_name + "/train.txt"
    train_data = pickle.load(open(train_filename, "rb"))
    test_filename = "datasets/" + dataset_name + "/test.txt"
    test_data = pickle.load(open(test_filename, "rb"))

    item_set = set()
    for items in train_data[0]:
        for id_ in range(len(items)):
            item_set.add(items[id_])
    for item in train_data[1]:
        item_set.add(item)
    for items in test_data[0]:
        for id_ in range(len(items)):
            item_set.add(items[id_])
    for item in test_data[1]:
        item_set.add(item)
    num_of_items = len(item_set)

    item_list = sorted(list(item_set))
    item_dict = dict()
    for i in range(1, len(item_set) + 1):
        item = item_list[i - 1]
        item_dict[item] = i

    train_data_x = list()
    train_data_y = list()
    test_data_x = list()
    test_data_y = list()
    for items in train_data[0]:
        new_list = []
        for item in items:
            new_item = item_dict[item]
            new_list.append(new_item)
        train_data_x.append(new_list)
    for item in train_data[1]:
        new_item = item_dict[item]
        train_data_y.append(new_item)
    for items in test_data[0]:
        new_list = []
        for item in items:
            new_item = item_dict[item]
            new_list.append(new_item)
        test_data_x.append(new_list)
    for item in test_data[1]:
        new_item = item_dict[item]
        test_data_y.append(new_item)

    max_length = 0
    for sample in train_data_x:
        max_length = len(sample) if len(sample) > max_length else max_length
    for sample in test_data_x:
        max_length = len(sample) if len(sample) > max_length else max_length

    train_seqs = np.zeros((len(train_data_x), max_length))
    train_poses = np.zeros((len(train_data_x), max_length))
    test_seqs = np.zeros((len(test_data_x), max_length))
    test_poses = np.zeros((len(test_data_x), max_length))
    for i in range(len(train_data_x)):
        seq = train_data_x[i]
        length = len(seq)
        pos = [j + 1 for j in range(length)]
        train_seqs[i][-length:] = seq
        train_poses[i][-length:] = pos

    for i in range(len(test_data_x)):
        seq = test_data_x[i]
        length = len(seq)
        pos = [j + 1 for j in range(length)]
        test_seqs[i][-length:] = seq
        test_poses[i][-length:] = pos

    train_target = np.array(train_data_y)
    test_target = np.array(test_data_y)
    train_data_set = (train_seqs, train_poses, train_target)
    test_data_set = (test_seqs, test_poses, test_target)
    return max_length, num_of_items, train_data_set, test_data_set


def get_data_loader(data_set, batch_size, shuffle):
    seqs, poses, targets = data_set

    seqs = torch.Tensor(seqs).long()
    poses = torch.Tensor(poses).long()
    targets = torch.Tensor(targets).long()

    seqs.requires_grad = False
    poses.requires_grad = False
    targets.requires_grad = False

    data_sets = TensorDataset(seqs, poses, targets)
    data_loader = DataLoader(data_sets, batch_size=batch_size, shuffle=shuffle)

    return data_loader, len(data_sets), targets


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )
        self.optional = self._action_groups.pop()
        self.required = self.add_argument_group("required arguments")
        self._action_groups.append(self.optional)

    def add_argument(self, *args, **kwargs):
        if kwargs.get("required", False):
            return self.required.add_argument(*args, **kwargs)
        else:
            return super().add_argument(*args, **kwargs)
