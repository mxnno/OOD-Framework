import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from datasets import load_metric
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils_DNNC import convert_examples_to_features, get_eval_dataloader
from utils.utils import get_result_path
import csv
import os


def evaluate_metriken_ohne_Treshold(args, scores):

    #Alle mit _ocsvm
    #evtl noch gda
    if args.model_ID == 14:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'maha_score']
    else:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score', 'gda_eucl_score', 'gda_maha_score', 'varianz_score']

    header = ['Method', "tnr_at_tpr95", "auroc", "dtacc", "au_in", "au_out"]
    csvPath = get_result_path(args)
    if not os.path.isdir(get_result_path(args)):
        os.mkdir(get_result_path(args))
    csvPath += "/results_metriken_ohne_Treshold.csv"

    with open(csvPath, 'w', encoding='utf-8') as csvf:

        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(header)

        for score_name in score_list:
            print
            data = []
            # ohne OCSVM
            pred_in = scores.__dict__[score_name + "_in"]
            pred_out = scores.__dict__[score_name + "_out"]
            tnr_at_tpr95, auroc, dtacc, au_in, au_out = get_metriken_ohne_Treshold(pred_in, pred_out)

            data = [tnr_at_tpr95, auroc, dtacc, au_in, au_out]

            writer.writerow([score_name.replace("_score", ""),] + data)

            # mit OCSVM 
            pred_in = scores.__dict__[score_name + "_in_ocsvm"]
            pred_out = scores.__dict__[score_name + "_out_ocsvm"]
            tnr_at_tpr95, auroc, dtacc, au_in, au_out = get_metriken_ohne_Treshold(pred_in, pred_out)

            data = [tnr_at_tpr95, auroc, dtacc, au_in, au_out]

            writer.writerow([score_name.replace("_score", "_ocsvm"),] + data)

    print("Result file saved at: " + csvPath)




def evaluate_mit_Treshold(args, scores, name):

    if args.model_ID == 14:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'maha_score']
    else:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'energy_score', 'entropy_score', 'gda_eucl_score', 'gda_maha_score', 'maha_score', 'varianz_score']

    header = ['Method', "in_acc + out_recall", "in_acc", "in_recall", "in_f1", "out_acc", "out_recall", "out_f1", "acc", "recall", "f1", "roc_auc", "fpr_95"]
    csvPath = get_result_path(args)
    if not os.path.isdir(get_result_path(args)):
        os.mkdir(get_result_path(args))
    csvPath += "/results_mit_Treshold_" + name + ".csv"
    with open(csvPath, 'w', encoding='utf-8') as csvf:

        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(header)

        for score_name in score_list:
            data = []
            y_pred_in = scores.__dict__[score_name + "_in"]
            y_pred_out = scores.__dict__[score_name + "_out"]

            labels_in = np.ones_like(y_pred_in).astype(np.int64)
            labels_out = np.zeros_like(y_pred_out).astype(np.int64)
            preds_gesamt = np.concatenate((y_pred_in, y_pred_out), axis=-1)
            labels_gesamt = np.concatenate((labels_in, labels_out), axis=-1)

            in_acc = accuracy_score(labels_in, y_pred_in)
            in_recall = recall_score(labels_in, y_pred_in)
            in_f1 = f1_score(labels_in, y_pred_in)
            out_acc = accuracy_score(labels_out, y_pred_out)
            out_recall = recall_score(labels_out, y_pred_out, pos_label=0)
            out_f1 = f1_score(labels_out, y_pred_out, pos_label=0)

            acc = accuracy_score(labels_gesamt, preds_gesamt)
            recall = recall_score(labels_gesamt, preds_gesamt)
            f1 = f1_score(labels_gesamt, preds_gesamt)
            roc_auc = roc_auc_score(labels_gesamt, preds_gesamt)
            fpr_95 = get_fpr_95(labels_gesamt, preds_gesamt)

            data = [(in_acc + out_recall), in_acc, in_recall, in_f1, out_acc, out_recall, out_f1, acc, recall, f1, roc_auc, fpr_95]

            writer.writerow([score_name.replace("_score", ""),] + data)

    print("Result file saved at: " + csvPath)


def evaluate_scores_ohne_Treshold(args, scores):

    if args.model_ID == 14:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp','maha_score']
    else:
        score_list = ['lof_score', 'doc_score', 'logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score', 'gda_eucl_score', 'gda_maha_score', 'varianz_score']

    #score_list = ['lof_score', 'doc_score', 'logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score', 'gda_eucl_score', 'gda_maha_score', 'varianz_score']

    header = ['Method', "in_acc + out_recall", "in_acc", "in_recall", "in_f1", "out_acc", "out_recall", "out_f1", "acc", "recall", "f1", "roc_auc", "fpr_95"]
    csvPath = get_result_path(args)
    if not os.path.isdir(get_result_path(args)):
        os.mkdir(get_result_path(args))
    csvPath += "/results_scores_ohne_Treshold.csv"
    with open(csvPath, 'w', encoding='utf-8') as csvf:

        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(header)

        for score_name in score_list:

            data = []
            if score_name in ['adb_score', 'lof_score', 'doc_score']:
                y_pred_in = scores.__dict__[score_name + "_in"]
                y_pred_out = scores.__dict__[score_name + "_out"]
                label_name = score_name.replace("_score", "")

            else:
                y_pred_in = scores.__dict__[score_name + "_in_ocsvm"]
                y_pred_out = scores.__dict__[score_name + "_out_ocsvm"]
                label_name = score_name.replace("_score", "_ocsvm_01")


            labels_in = np.ones_like(y_pred_in).astype(np.int64)
            labels_out = np.zeros_like(y_pred_out).astype(np.int64)
            preds_gesamt = np.concatenate((y_pred_in, y_pred_out), axis=-1)
            labels_gesamt = np.concatenate((labels_in, labels_out), axis=-1)

            in_acc = accuracy_score(labels_in, y_pred_in)
            in_recall = recall_score(labels_in, y_pred_in)
            in_f1 = f1_score(labels_in, y_pred_in)
            out_acc = accuracy_score(labels_out, y_pred_out)
            out_recall = recall_score(labels_out, y_pred_out, pos_label=0)
            out_f1 = f1_score(labels_out, y_pred_out, pos_label=0)

            acc = accuracy_score(labels_gesamt, preds_gesamt)
            recall = recall_score(labels_gesamt, preds_gesamt)
            f1 = f1_score(labels_gesamt, preds_gesamt)
            roc_auc = roc_auc_score(labels_gesamt, preds_gesamt)
            fpr_95 = get_fpr_95(labels_gesamt, preds_gesamt)

            data = [(in_acc + out_recall), in_acc, in_recall, in_f1, out_acc, out_recall, out_f1, acc, recall, f1, roc_auc, fpr_95]

            writer.writerow([label_name,] + data)

    print("Result file saved at: " + csvPath)


def evaluate_NLI(args, y_pred_in, y_pred_out):

    header = ['Method', "in_acc + out_recall", "in_acc", "in_recall", "in_f1", "out_acc", "out_recall", "out_f1", "acc", "recall", "f1", "roc_auc", "fpr_95"]

    labels_in = np.ones_like(y_pred_in).astype(np.int64)
    labels_out = np.zeros_like(y_pred_out).astype(np.int64)
    preds_gesamt = np.concatenate((y_pred_in, y_pred_out), axis=-1)
    labels_gesamt = np.concatenate((labels_in, labels_out), axis=-1)

    csvPath = get_result_path(args)
    if not os.path.isdir(get_result_path(args)):
        os.mkdir(get_result_path(args))
    csvPath += "/results_NLI.csv"
    with open(csvPath, 'w', encoding='utf-8') as csvf:

        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(header)


        in_acc = accuracy_score(labels_in, y_pred_in)
        in_recall = recall_score(labels_in, y_pred_in)
        in_f1 = f1_score(labels_in, y_pred_in)
        out_acc = accuracy_score(labels_out, y_pred_out)
        out_recall = recall_score(labels_out, y_pred_out, pos_label=0)
        out_f1 = f1_score(labels_out, y_pred_out, pos_label=0)

        acc = accuracy_score(labels_gesamt, preds_gesamt)
        recall = recall_score(labels_gesamt, preds_gesamt)
        f1 = f1_score(labels_gesamt, preds_gesamt)
        roc_auc = roc_auc_score(labels_gesamt, preds_gesamt)
        fpr_95 = get_fpr_95(labels_gesamt, preds_gesamt)

        data = [(in_acc + out_recall), in_acc, in_recall, in_f1, out_acc, out_recall, out_f1, acc, recall, f1, roc_auc, fpr_95]

        writer.writerow(["NLI", ] + data)
    
    print("Result file saved at: " + csvPath)




def evaluate_ADB(args, y_pred_in, y_pred_out):
    header = ['Method', "in_acc + out_recall", "in_acc", "in_recall", "in_f1", "out_acc", "out_recall", "out_f1", "acc", "recall", "f1", "roc_auc", "fpr_95"]

    labels_in = np.ones_like(y_pred_in).astype(np.int64)
    labels_out = np.zeros_like(y_pred_out).astype(np.int64)
    preds_gesamt = np.concatenate((y_pred_in, y_pred_out), axis=-1)
    labels_gesamt = np.concatenate((labels_in, labels_out), axis=-1)

    csvPath = get_result_path(args)
    if not os.path.isdir(get_result_path(args)):
        os.mkdir(get_result_path(args))
    csvPath += "/results_ADB.csv"
    with open(csvPath, 'w', encoding='utf-8') as csvf:

        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(header)


        in_acc = accuracy_score(labels_in, y_pred_in)
        in_recall = recall_score(labels_in, y_pred_in)
        in_f1 = f1_score(labels_in, y_pred_in)
        out_acc = accuracy_score(labels_out, y_pred_out)
        out_recall = recall_score(labels_out, y_pred_out, pos_label=0)
        out_f1 = f1_score(labels_out, y_pred_out, pos_label=0)

        acc = accuracy_score(labels_gesamt, preds_gesamt)
        recall = recall_score(labels_gesamt, preds_gesamt)
        f1 = f1_score(labels_gesamt, preds_gesamt)
        roc_auc = roc_auc_score(labels_gesamt, preds_gesamt)
        fpr_95 = get_fpr_95(labels_gesamt, preds_gesamt)

        data = [(in_acc + out_recall), in_acc, in_recall, in_f1, out_acc, out_recall, out_f1, acc, recall, f1, roc_auc, fpr_95]

        writer.writerow(["ADB", ] + data)
    
    print("Result file saved at: " + csvPath)
##############################################################################################################################################################


def evaluate(args, model, eval_dataset, tag=""):
    
    #F1
    metric = load_metric("accuracy")

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels, )
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result


    label_list, logit_list = [], []
    for batch in tqdm(eval_dataset):
        model.eval()
        
        if args.model_ID == 8:
            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataset, desc="Evaluating"):
                input_ids = input_ids.to(args.device)
                input_mask = input_mask.to(args.device)
                segment_ids = segment_ids.to(args.device)

            outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            labels = label_ids.numpy()
        else:
            labels = batch["labels"].detach().cpu().numpy()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            batch["labels"] = None
            outputs = model(**batch)

        logits_2 = outputs[0]
        logits = logits_2.detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)


    preds = np.concatenate(logit_list, axis=0)

    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)


    preds1 = np.argmax(preds, axis=1)
    acc, f1 = get_acc_and_f1(preds1, labels, model.num_labels)
    print("acc_" + tag + ": " + str(acc))
    print("f1_" + tag + ": " + str(f1))

    results = {"accuracy_" + tag: acc, "f1_" + tag: f1}
    return results

def evaluate_DNNC(args, model, tokenizer, eval_dataset):

    if len(eval_dataset) == 0:
            return None

    eval_features = convert_examples_to_features(args, eval_dataset, tokenizer, train = False)
    eval_dataloader = get_eval_dataloader(eval_features, args.batch_size)

    return evaluate(args, model, eval_dataloader,"dev")
    
    # model.eval()
    # eval_accuracy = 0
    # nb_eval_examples = 0

    # for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    #     input_ids = input_ids.to(args.device)
    #     input_mask = input_mask.to(args.device)
    #     segment_ids = segment_ids.to(args.device)

    #     with torch.no_grad():
    #         outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
    #         logits = outputs[0]

    #     logits = logits.detach().cpu().numpy()
    #     label_ids = label_ids.numpy()
    #     tmp_eval_accuracy = get_accuracy_DNNC(logits, label_ids)

    #     eval_accuracy += tmp_eval_accuracy
    #     nb_eval_examples += input_ids.size(0)

    # eval_accuracy = eval_accuracy / nb_eval_examples
    # return {"accuracy": eval_accuracy}
    


def get_accuracy(preds, labels):
    return (preds == labels).mean().item()

def get_accuracy_DNNC(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

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
    

def get_metriken_ohne_Treshold (scores_in, scores_out):
    #scores_in: 450 1D Scores von trainingsdaten

    known = scores_in
    novel = scores_out
    known.sort()
    novel.sort()
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    num_k = known.shape[0]
    num_n = novel.shape[0]
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n

    
    tpr =  np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])



    #######################################################
    #Area Under the Receiver Operating Characteristic Curve
    auroc = -np.trapz(1.-fpr, tpr)
    #######################################################
    

    #######################################################
    #Detection Accuracy
    dtacc = .5 * (tp/tp[0] + 1.-fp/fp[0]).max()
    #######################################################


    #######################################################
    #Area under the Precision-Recall curve
    #für ID: Auin
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    auin = -np.trapz(pin[pin_ind], tpr[pin_ind])

    #für OOD: Auout
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    auout = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
    #######################################################



    return tnr_at_tpr95, auroc, dtacc, auin, auout



