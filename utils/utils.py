import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

task_to_labels = {
    'full': 151,
    'unlabeled': 2,
    'zero': 2
}
def get_num_labels(args):
    if args.ood_data == 'zero':
        return task_to_labels[args.id_data]
    else:
        return task_to_labels[args.id_data] - 1

def save_model(model, path):
    model.save_pretrained(path)
    print("Model saved at: " + path)

def save_tensor(tensor, path, tesnor_name="Tensor"):
    torch.save(tensor, path)
    print(tesnor_name + " saved at: " + path)



# def collate_fn(batch):
#     max_len = max([len(f["input_ids"]) for f in batch])
#     input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
#     input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
#     labels = [f["labels"] for f in batch]
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     input_mask = torch.tensor(input_mask, dtype=torch.float)
#     labels = torch.tensor(labels, dtype=torch.long)
#     outputs = {
#         "input_ids": input_ids,
#         "attention_mask": input_mask,
#         "labels": labels,
#     }
#     return outputs
