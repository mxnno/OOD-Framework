import torch
import numpy as np
from evaluation import get_auroc, get_fpr_95
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from numpy import savetxt
from utils.utils import get_labels, get_num_labels


def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict

def detect_ood(args, model, prepare_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None):
    
    #Varianz für Distanzen bestimmen

    #idee: über  cm = confusion_matrix(y_true, y_pred)
    #binär tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # und dann classification_report

    model.prepare_ood(prepare_dataset)

    if centroids is not None:
        keys = ['softmax', 'maha', 'cosine', 'energy', 'adb']
    else:
        keys = ['softmax', 'maha', 'cosine', 'energy']

    in_scores = []
    for batch in tqdm(test_id_dataset):

        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids=centroids, delta=delta)
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, keys)

    out_scores = []
    for batch in tqdm(test_ood_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids=centroids, delta=delta)
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

    wandb.log(outputs) if args.wandb == "log" else None


def test_detect_ood(args, model, prepare_dataset, test_dataset, centroids=None, delta=None):
    
    #Varianz für Distanzen bestimmen

    #idee: über  cm = confusion_matrix(y_true, y_pred)
    #binär tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # und dann classification_report

    model.prepare_ood(prepare_dataset)

    if centroids is not None:
        keys = ['softmax', 'maha', 'cosine', 'energy', 'adb']
    else:
        keys = ['softmax', 'maha', 'cosine', 'energy']

    in_scores = []
    total_labels = torch.empty(0,dtype=torch.long).to(args.device)
    total_preds = torch.empty(0,dtype=torch.long).to(args.device)
    for batch in tqdm(test_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        label_ids = batch["labels"]
        with torch.no_grad():
            ood_keys, logits, preds = model.compute_ood(**batch, centroids=centroids, delta=delta, test=True)
            
            softmax_score = ood_keys['softmax']

            total_labels = torch.cat((total_labels,label_ids))
            total_preds = torch.cat((total_preds, preds))

    y_pred = total_preds.cpu().numpy()
    y_true = total_labels.cpu().numpy()

    
    labels = list(range(0, get_num_labels(args)))

    cm = confusion_matrix(y_true, y_pred)

    # save numpy array as csv file
    savetxt('confusion_matrix.csv', cm, delimiter=',')

    #Classfication Report
    print(classification_report(y_true, y_pred, labels=labels))