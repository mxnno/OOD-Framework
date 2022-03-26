import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from datasets import load_metric
from tqdm import tqdm
import torch.nn.functional as F



def evaluate(args, model, eval_dataset, tag="train"):
    
    #Accuracy + F1
    metric = load_metric("accuracy")

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels, )
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result


    label_list, logit_list = [], []
    preds2_list = []
    for batch in tqdm(eval_dataset):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["labels"] = None
        outputs = model(**batch)

        logits_2 = outputs[0]
        logits = logits_2.detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
        softmax_label = F.softmax(logits_2, dim=-1).max(-1)[1]
        preds2_list.append(softmax_label.detach().cpu().numpy())


    preds = np.concatenate(logit_list, axis=0)
    preds2 = np.concatenate(preds2_list, axis=0)

    labels = np.concatenate(label_list, axis=0)
    #results = compute_metrics(preds, labels)
    print("Accuracy: " + str(results))

    #index of max
    #bei multiclass evtl noch ein softmax?
    preds1 = np.argmax(preds, axis=1)
    acc, f1 = get_acc_and_f1(preds1, labels, model.num_labels)
    print("acc " + str(acc))
    print("f1 " + str(f1))

    acc2, f2 = get_acc_and_f1(preds2, labels, model.num_labels)
    print("acc2 " + str(acc2))
    print("f12 " + str(f2))
    results = {"accuracy": acc, "f1": f1}
    return results


def get_accuracy(preds, labels):
    return (preds == labels).mean().item()

def get_acc_and_f1(preds, labels, num_labels):
    acc = get_accuracy(preds, labels)

    average = 'binary'
    if num_labels > 2:
        average = 'macro'
    f1 = f1_score(y_true=labels, y_pred=preds, average=average).item()
    return acc, f1

def get_auroc(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    return roc_auc_score(new_key, prediction)


def get_fpr_95(key, prediction):
    new_key = np.copy(key)
    new_key[key == 0] = 0
    new_key[key > 0] = 1
    score = fpr_and_fdr_at_recall(new_key, prediction)
    return score


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))
