import os
import torch
from utils.instance import Instance
from utils.vocab import Vocab

DATA_NAMES = ["apparel", "baby", "books", "camera_photo", "dvd",
             "electronics", "health_personal_care", "imdb",
             "kitchen_housewares", "magazines", "MR", "music",
             "software", "sports_outdoors", "toys_games", "video"]


def get_task(task_id):
    return DATA_NAMES[task_id]


def load_dataset(dir_path, mode='train'):
    # mode: train / test / unlabel
    assert os.path.exists(dir_path) and os.path.isdir(dir_path) and mode in ['train', 'test', 'unlabel']
    for task_id, data_name in enumerate(DATA_NAMES):
        fn = os.path.join(dir_path, f'{data_name}.task.{mode}')
        yield _load_data(fn, task_id)


def _load_data(file_name, task_id):
    assert os.path.exists(file_name)
    dataset = []
    with open(file_name, 'r', encoding='utf-8') as fin:
        reader = map(lambda x: x.strip().split('\t'), fin)
        for item in reader:
            if len(item) == 2:
                lbl, data_str = item
                dataset.append(Instance(task_id,
                                        data_str.split(' '),
                                        int(lbl)))
    return dataset


# create vocab according to the dataset of all tasks
def create_vocab(all_data, min_count=3):
    wd_vocab = Vocab(min_count=min_count, bos=None, eos=None)
    for task_data in all_data:
        for inst in task_data:
            wd_vocab.add(inst.data)
    return wd_vocab


def batch_variable(batch_data, wd_vocab):
    bs = len(batch_data)
    max_len = max(len(inst.data) for inst in batch_data)

    task_ids = torch.zeros((bs, ), dtype=torch.long)
    wd_ids = torch.zeros((bs, max_len), dtype=torch.long)
    lbl_ids = torch.zeros((bs, ), dtype=torch.long)

    for i, inst in enumerate(batch_data):
        task_ids[i] = torch.tensor(inst.task)
        wd_ids[i, :len(inst.data)] = torch.tensor(wd_vocab.inst2idx(inst.data))
        lbl_ids[i] = torch.tensor(inst.label)

    return Batch(task_ids, wd_ids, lbl_ids)


class Batch(object):
    def __init__(self, task_ids, wd_ids, lbl_ids):
        self.task_ids = task_ids
        self.wd_ids = wd_ids
        self.lbl_ids = lbl_ids

    def to_device(self, device):
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))


































