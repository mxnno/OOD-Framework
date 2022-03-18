import torch
import numpy as np
from evaluation import get_auroc, get_fpr_95
import wandb

def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict

def detect_ood(args, model, prepare_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None):
    
    #Varianz für Distanzen bestimmen
    model.prepare_ood(prepare_dataset)

    keys = ['softmax', 'maha', 'cosine', 'energy', 'adb']

    in_scores = []
    for batch in test_id_dataset:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids)
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, keys)
    print(in_scores)

    out_scores = []
    for batch in test_ood_dataset:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch)
            out_scores.append(ood_keys)
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:
        ins = np.array(in_scores[key], dtype=np.float64)
        outs = np.array(out_scores[key], dtype=np.float64)
        inl = np.ones_like(ins).astype(np.int64)
        outl = np.zeros_like(outs).astype(np.int64)
        scores = np.concatenate([ins, outs], axis=0)
        labels = np.concatenate([inl, outl], axis=0)

        auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

        outputs[tag + "_" + key + "_auroc"] = auroc
        outputs[tag + "_" + key + "_fpr95"] = fpr_95

    wandb.log(outputs)