import time
import random
import numpy as np
import torch
from modules.optimizer import Optimizer
from model.ATClassifier import ATClassifier
from config.conf import args_config, data_config
from utils.datautil import get_task, load_dataset, create_vocab, batch_variable
import torch.nn.functional as F
from logger.logger import logger
from utils.dataset import MyDataSet, DataLoader, data_split
import torch.nn.utils as nn_utils


class Trainer(object):
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path
        # the dataset for all tasks
        self.train_set, self.dev_set, self.test_set = self.build_dataset(data_path['dataset'])
        self.args.nb_task = len(self.train_set)
        all_data = self.train_set.data + self.dev_set.data + self.test_set.data
        self.wd_vocab = self.build_vocab(all_data)
        self.model = self.build_model(args, self.wd_vocab)

    def build_dataset(self, data_path):
        train_data = []
        test_data = []
        for data in load_dataset(data_path, 'train'):
            train_data.append(data)

        for data in load_dataset(data_path, 'test'):
            test_data.append(data)

        train_set, dev_set, test_set = [], [], []
        for task_id, (task_train_data, task_test_data) in enumerate(zip(train_data, test_data)):
            print(f'loading dataset {get_task(task_id)} ...')
            train_part, dev_part, test_part = data_split(task_train_data + task_test_data,
                                                         split_rate=[7, 2, 1],
                                                         shuffle=True)
            train_set.append(train_part)
            dev_set.append(dev_part)
            test_set.append(test_part)

        return MyDataSet(train_set), MyDataSet(dev_set), MyDataSet(test_set)

    def build_vocab(self, all_data: list, min_count=3):
        wd_vocab = create_vocab(all_data, min_count)
        print(f'Word vocabulary is built, total size is {len(wd_vocab)}.')
        return wd_vocab

    def build_model(self, args, wd_vocab):
        embed_count = wd_vocab.load_embeddings(data_path['embed_weights'])
        print("%d pre-trained embeddings loaded ..." % embed_count)
        args.num_embeddings = len(wd_vocab)
        model = ATClassifier(args, pre_embed=wd_vocab.embeddings).to(args.device)
        print(model)
        return model

    def train_eval(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = Optimizer(params, args)
        task_dev_acc_dict = dict()
        task_test_err_dict = dict()
        for ep in range(1, self.args.epoch+1):
            for task_id, train_data in enumerate(self.train_set):
                task_name = get_task(task_id)
                print(f'training {task_name} task ...')
                task_train_loss, task_train_acc = self.train_iter(ep, task_id, train_data, optimizer)

                task_dev_acc = self.eval(task_id, self.dev_set[task_id])
                if task_id not in task_dev_acc_dict or task_dev_acc_dict[task_id] < task_dev_acc:
                    task_dev_acc_dict[task_id] = task_dev_acc

                    task_test_acc = self.eval(task_id, self.test_set[task_id])
                    task_test_err_dict[task_id] = 1 - task_test_acc

                logger.info('[Epoch %d][Task %s] train loss: %.4f, lr: %f, Train ACC: %.4f, Dev ACC: %.4f, Best Dev ACC: %.4f, Best Test ERR: %.4f' % (
                        ep, task_name, task_train_loss, optimizer.get_lr(), task_train_acc, task_dev_acc, task_dev_acc_dict[task_id], task_test_err_dict[task_id]))

            for tid, test_err in task_test_err_dict.items():
                logger.info('[Epoch %d][Task %s] Test Err: %.4f' % (ep, get_task(tid), test_err))

        all_task_err = list(task_test_err_dict.values())
        logger.info('Avg Test Err: %.4f' % np.mean(all_task_err))

    def lambda_(self, cur_step, total_step):
        p = cur_step / total_step
        return 2 / (1 + np.exp(-10 * p)) - 1

    def train_iter(self, ep, task_id, train_data, optimizer):
        t1 = time.time()
        train_acc, train_loss = 0., 0.
        self.model.train()
        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)
        total_step = 200 * len(train_loader)
        step = 0
        for i, batcher in enumerate(train_loader):
            batch = batch_variable(batcher, self.wd_vocab)
            batch.to_device(self.args.device)
            adv_lmbd = self.lambda_(step, total_step)
            task_logits, share_logits = self.model(task_id, batch.wd_ids, adv_lmbd)
            loss_task = F.cross_entropy(task_logits, batch.lbl_ids)
            loss_share = F.cross_entropy(share_logits, batch.task_ids)
            loss = loss_task + self.args.adv_loss_w * loss_share
            loss.backward()
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=args.grad_clip)
            optimizer.step()
            self.model.zero_grad()

            loss_val = loss.data.item()
            train_loss += loss_val
            train_acc += (task_logits.data.argmax(dim=-1) == batch.lbl_ids).sum().item()
            logger.info('[Epoch %d][Task %s] Iter%d time cost: %.2fs, lr: %.6f, train acc: %.4f, train loss: %.4f' % (
                ep, get_task(task_id), i + 1, (time.time() - t1), optimizer.get_lr(), train_acc/len(train_data), loss_val))

            step += 1

        return train_loss/len(train_data), train_acc/len(train_data)

    def eval(self, task_id, test_data):
        print(f'evaluating {get_task(task_id)} task ...')
        nb_correct, nb_total = 0, 0
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size)
        with torch.no_grad():
            for i, batcher in enumerate(test_loader):
                batch = batch_variable(batcher, self.wd_vocab)
                batch.to_device(self.args.device)
                task_logits, share_logits = self.model(task_id, batch.wd_ids)
                nb_correct += (task_logits.data.argmax(dim=-1) == batch.lbl_ids).sum().item()
                nb_total += len(batch.lbl_ids)
        acc = nb_correct / nb_total
        # err = 1 - acc
        return acc


if __name__ == '__main__':
    random.seed(1347)
    np.random.seed(2343)
    torch.manual_seed(1453)
    torch.cuda.manual_seed(1347)
    torch.cuda.manual_seed_all(1453)
    print('cuda available:', torch.cuda.is_available())
    print('cuDNN available:', torch.backends.cudnn.enabled)
    print('gpu numbers:', torch.cuda.device_count())

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
        torch.cuda.empty_cache()
    else:
        args.device = torch.device('cpu')

    data_path = data_config('./config/data_path.json')
    trainer = Trainer(args, data_path)
    trainer.train_eval()
