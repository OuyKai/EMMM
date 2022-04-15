import os
import logging

import torch

from mlt import REATTN
from metric import get_mrr, get_recall
from utils import ArgumentParser, data_reader, get_data_loader, str_time

# set_seed(2023)
parser = ArgumentParser()
parser.add_argument(
    '--device', type=int, default=0, help='the index of GPU device (-1 for CPU)'
)
parser.add_argument('--batch-size', type=int, default=100, help='the batch size')
parser.add_argument(
    '--embedding-dim', type=int, default=100, help='the dimensionality of embeddings'
)
parser.add_argument(
    '--log-level',
    choices=['debug', 'info', 'warning', 'error'],
    default='debug',
    help='the log level',
)
args = parser.parse_args()
log_level = getattr(logging, args.log_level.upper(), None)
logging.basicConfig(format='%(message)s', level=log_level)
# logging.debug(args)
# logging.info(f'PID is {os.getpid()}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.device)

best_models = os.listdir("save/")
dataset_names = ["Tmall", "RetailRocket", "diginetica"]
for dataset_name in dataset_names:
    logging.info(f'\nreading dataset {dataset_name}...')
    max_length, num_of_items, train_data_set, test_data_set = data_reader(dataset_name)
    test_dataload, _, test_y = get_data_loader(test_data_set, args.batch_size, False)

    for bm in best_models:
        if dataset_name not in bm:
            continue
        print("\nBest metric of this model is " + bm.split("_")[-1][:-4])
        print(bm)
        bm = os.path.join("save/", bm)
        model = REATTN(args.embedding_dim, num_of_items + 1, max_length + 2).cuda()
        model.load_state_dict(torch.load(bm))
        model.eval()
        with torch.no_grad():
            y_pre_all = torch.LongTensor().cuda()
            y_pre_all_10 = torch.LongTensor()
            y_pre_all_5 = torch.LongTensor()
            for x_test, pos_test, y_test in test_dataload:
                with torch.no_grad():
                    # y_pre = model.predict(x_test.cuda(), pos_test.cuda(), 20)
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

            print("[%s] " % str_time() + "Recall@20: " + "%.4f" % recall + " Recall@10: " + "%.4f" % recall_10 + "  Recall@5: " + "%.4f" % recall_5)
            print("[%s] " % str_time() + "MRR@20: " + "%.4f" % mrr.tolist() + " MRR@10: " + "%.4f" % mrr_10.tolist() + " MRR@5: " + "%.4f" % mrr_5.tolist())
