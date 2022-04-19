import os
import logging

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from mlt import REATTN
from metric import get_mrr, get_recall
from utils import ArgumentParser, data_reader, get_data_loader, set_seed, str_time
PID = os.getpid()
set_seed(2023)
parser = ArgumentParser()
parser.add_argument(
    '--device', type=int, default=0, help='the index of GPU device (-1 for CPU).'
)
parser.add_argument(
    '--model-drop', type=float, default=0.5, help='the dropout ratio for model.'
)
parser.add_argument(
    '--coefficient', type=int, default=20, help='the coefficient for model.'
)
parser.add_argument(
    '--dataset-name', type=str, default="Tmall", help='the dataset set directory.'
)
parser.add_argument('--batch-size', type=int, default=100, help='the batch size.')
parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate.')
parser.add_argument(
    '--epochs', type=int, default=50, help='the maximum number of training epochs.'
)
parser.add_argument(
    '--embedding-dim', type=int, default=100, help='the dimensionality of embeddings.'
)
parser.add_argument(
    '--weight-decay',
    type=float,
    default=1e-5,
    help='the weight_decay of opt.',
)
parser.add_argument(
    '--log-level',
    choices=['debug', 'info', 'warning', 'error'],
    default='debug',
    help='the log level.',
)
args = parser.parse_args()
log_level = getattr(logging, args.log_level.upper(), None)
logging.basicConfig(format='%(message)s', level=log_level)
logging.debug(args)
logging.info(f'PID is {os.getpid()}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.device)

logging.info(f'\nreading dataset {args.dataset_name}...\n')
model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '.pth'
max_length, num_of_items, train_data_set, test_data_set = data_reader(args.dataset_name)
train_dataload, train_data_size, _ = get_data_loader(train_data_set, args.batch_size, True)
test_dataload, _, test_y = get_data_loader(test_data_set, args.batch_size, False)

criterion = nn.CrossEntropyLoss().cuda()
T = 0.5
train_batch = len(train_dataload)
avg_cost = np.zeros([args.epochs, 3])  # 0\1\2 train loss
lambda_weight = np.ones([3, args.epochs])
weight = np.zeros((args.epochs, 3))
model = REATTN(args.embedding_dim, num_of_items + 1, max_length + 2, coefficient=args.coefficient, dropout=args.model_drop).cuda()
logging.debug(model)
opti = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
best_result = 0
best_result_ = []
global_beat_result = {"Recall@20": 0, "Recall@10": 0, "Recall@5": 0, "MRR@20": 0, "MRR@10": 0, "MRR@5": 0}
for epoch in range(args.epochs):
    losses = 0
    if epoch == 0 or epoch == 1:
        lambda_weight[:, epoch] = 1.0
    else:
        w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
        w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
        w_3 = avg_cost[epoch - 1, 2] / avg_cost[epoch - 2, 2]
        lambda_weight[0, epoch] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
        lambda_weight[1, epoch] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
        lambda_weight[2, epoch] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

    model.train()
    for step, (x_train, pos_train, y_train) in enumerate(train_dataload):
        opti.zero_grad()
        cost = np.zeros(3, dtype=np.float32)
        q0, q1, q2, q = model(x_train.cuda(), pos_train.cuda())

        loss_main = criterion(q, y_train.cuda() - 1)
        loss0 = criterion(q0, y_train.cuda() - 1)
        loss1 = criterion(q1, y_train.cuda() - 1)
        loss2 = criterion(q2, y_train.cuda() - 1)
        cost[0] = loss0.item()
        cost[1] = loss1.item()
        cost[2] = loss2.item()
        avg_cost[epoch, :3] += cost[:3] / train_batch
        loss = loss_main + lambda_weight[0, epoch] * loss0 + lambda_weight[1, epoch] * loss1 + lambda_weight[2, epoch] * loss2
        loss.backward()
        opti.step()
        losses += loss.item()
        if (step + 1) % 100 == 0:
            print("[%s] [%03d/%03d] [%04d/%04d] mean_loss < %0.2f >" % (str_time(), epoch, args.epochs, step, train_data_size / args.batch_size, losses / step + 1))
        weight[epoch, 0], weight[epoch, 1], weight[epoch, 2] = lambda_weight[0, epoch], lambda_weight[1, epoch], lambda_weight[2, epoch]

    model.eval()
    with torch.no_grad():
        y_pre_all = torch.LongTensor().cuda()
        y_pre_all_10 = torch.LongTensor()
        y_pre_all_5 = torch.LongTensor()
        for x_test, pos_test, y_test in test_dataload:
            with torch.no_grad():
                y_pre = model(x_test.cuda(), pos_test.cuda(), 20)
                y_pre_all = torch.cat((y_pre_all, y_pre), 0)
                y_pre_all_10 = torch.cat((y_pre_all_10, y_pre.cpu()[:, :10]), 0)
                y_pre_all_5 = torch.cat((y_pre_all_5, y_pre.cpu()[:, :5]), 0)
        recall = get_recall(y_pre_all, test_y.cuda().unsqueeze(1) - 1)
        recall_10 = get_recall(y_pre_all_10, test_y.unsqueeze(1) - 1)
        recall_5 = get_recall(y_pre_all_5, test_y.unsqueeze(1) - 1)
        mrr = get_mrr(y_pre_all, test_y.cuda().unsqueeze(1) - 1).tolist()
        mrr_10 = get_mrr(y_pre_all_10, test_y.unsqueeze(1) - 1).tolist()
        mrr_5 = get_mrr(y_pre_all_5, test_y.unsqueeze(1) - 1).tolist()

        print("[%s] " % str_time() + "Recall@20: " + "%.4f" % recall + "  Recall@10: " + "%.4f" % recall_10 + "  Recall@5: " + "%.4f" % recall_5)
        print("[%s] " % str_time() + "MRR@20: " + "%.4f" % mrr + "  MRR@10: " + "%.4f" % mrr_10 + "  MRR@5: " + "%.4f" % mrr_5)

        if mrr > global_beat_result["MRR@20"]:
            tmp_model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '_MRR@20' + '.pth'
            torch.save(model.state_dict(), tmp_model_save_path)
            global_beat_result["MRR@20"] = mrr
        if mrr_10 > global_beat_result["MRR@10"]:
            tmp_model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '_MRR@10' + '.pth'
            torch.save(model.state_dict(), tmp_model_save_path)
            global_beat_result["MRR@10"] = mrr_10
        if mrr_5 > global_beat_result["MRR@5"]:
            tmp_model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '_MRR@5' + '.pth'
            torch.save(model.state_dict(), tmp_model_save_path)
            global_beat_result["MRR@5"] = mrr_5
        if recall > global_beat_result["Recall@20"]:
            tmp_model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '_Recall@20' + '.pth'
            torch.save(model.state_dict(), tmp_model_save_path)
            global_beat_result["Recall@20"] = recall
        if recall_10 > global_beat_result["Recall@10"]:
            tmp_model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '_Recall@10' + '.pth'
            torch.save(model.state_dict(), tmp_model_save_path)
            global_beat_result["Recall@10"] = recall_10
        if recall_5 > global_beat_result["Recall@5"]:
            tmp_model_save_path = 'BestModel/best_' + str(args.dataset_name) + '_' + str(PID) + '_Recall@5' + '.pth'
            torch.save(model.state_dict(), tmp_model_save_path)
            global_beat_result["Recall@5"] = recall_5

        if best_result < recall:
            best_result = recall
            best_result_ = [recall_5, recall_10, recall, mrr_5, mrr_10, mrr]
            torch.save(model.state_dict(), model_save_path)

        print("[%s] " % str_time() + "best result: " + str(best_result))
        print("[%s] " % str_time() + "global best result: " + str(global_beat_result))

np.savetxt('lambda-weight-DWA.csv', weight, delimiter=',')

model = REATTN(args.embedding_dim, num_of_items + 1, max_length + 2, coefficient=args.coefficient).cuda()
model.load_state_dict(torch.load(model_save_path))
model.eval()
with torch.no_grad():
    y_pre_all = torch.LongTensor().cuda()
    y_pre_all_10 = torch.LongTensor()
    y_pre_all_5 = torch.LongTensor()
    for x_test, pos_test, y_test in test_dataload:
        with torch.no_grad():
            y_pre = model(x_test.cuda(), pos_test.cuda(), 20)
            y_pre_all = torch.cat((y_pre_all, y_pre), 0)
            y_pre_all_10 = torch.cat((y_pre_all_10, y_pre.cpu()[:, :10]), 0)
            y_pre_all_5 = torch.cat((y_pre_all_5, y_pre.cpu()[:, :5]), 0)
    recall = get_recall(y_pre_all, test_y.cuda().unsqueeze(1) - 1)
    recall_10 = get_recall(y_pre_all_10, test_y.unsqueeze(1) - 1)
    recall_5 = get_recall(y_pre_all_5, test_y.unsqueeze(1) - 1)
    mrr = get_mrr(y_pre_all, test_y.cuda().unsqueeze(1) - 1)
    mrr_10 = get_mrr(y_pre_all_10, test_y.unsqueeze(1) - 1)
    mrr_5 = get_mrr(y_pre_all_5, test_y.unsqueeze(1) - 1)

    print("\n[%s] " % str_time() + "best model")
    print("[%s] " % str_time() + "Recall@20: " + "%.4f" % recall + " Recall@10: " + "%.4f" % recall_10 + "  Recall@5: " + "%.4f" % recall_5)
    print("[%s] " % str_time() + "MRR@20: " + "%.4f" % mrr.tolist() + " MRR@10: " + "%.4f" % mrr_10.tolist() + " MRR@5: " + "%.4f" % mrr_5.tolist())

print("\n[%s] " % str_time() + "global best result")
print("[%s] " % str_time() + "Recall@20: " + "%.4f" % global_beat_result["Recall@20"] + " Recall@10: " + "%.4f" % global_beat_result["Recall@10"] + "  Recall@5: " + "%.4f" % global_beat_result["Recall@5"])
print("[%s] " % str_time() + "MRR@20: " + "%.4f" % global_beat_result["MRR@20"] + " MRR@10: " + "%.4f" % global_beat_result["MRR@10"] + " MRR@5: " + "%.4f" % global_beat_result["MRR@5"])
print("\n[%s] " % str_time() + "End and Best Wish ...\n")
