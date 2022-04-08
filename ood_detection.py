import torch
import numpy as np
from evaluation import get_auroc, get_fpr_95
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from numpy import savetxt
from utils.utils import get_labels, get_num_labels
from utils.utils_DNNC import *


#Was macht welches Paper?
#
#ohne OOD-Daten
#ID 1: sehr komisch -> auroc, AU-In, Au-Out, Acc-Out
#ID 2: Aufteilung Test ID/OOD, score berechnen -> auroc und fpr95 (Funktionen nicht übersichtlich)
#ID 8: Aufteilung Test ID/OOD, nur softmax score-> Acc-In, Recall-Out, Prec-Out, F1-Out (Funktionen übersichtlich)
#ID 9: Aufteilung Test ID/OOD, nur softmax -> Acc-In, Auroc-Out, Aupr-In-Out-Ges, FPR-Ges, TNR@95TPR-Out, mini_ood_error (so naja)
#I D14: keine Aufteilung, nutzen confusion matrix

def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict

def get_treshold(method, value):

    if method == "sofmtax":
        if value > 0.7:
            return 1
        else:
            return 0



def detect_ood(args, model, prepare_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None):
    
    #Varianz für Distanzen bestimmen

    #idee: über  cm = confusion_matrix(y_true, y_pred)
    #binär tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # und dann classification_report



    #OOD Detection + Eval in 3 Schritten
    # 1. Model prediction
    # 2. Threshold anwenden -> 0 oder 1
    # 3. Metriken berechnen

    model.prepare_ood(prepare_dataset)

    if centroids is not None:
        keys = ['softmax', 'maha', 'cosine', 'energy', 'adb']
    else:
        keys = ['softmax', 'maha', 'cosine', 'energy']

    in_scores = []
    i = 0
    for batch in tqdm(test_id_dataset):
        i +=1
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids=centroids, delta=delta)
            #for i, _ in enumerate(ood_keys["softmax"]):
            #    ood_keys["softmax"][i] = get_treshold("sofmtax", ood_keys["softmax"][i])
            in_scores.append(ood_keys)

        if i == 2:
            break

    in_scores = merge_keys(in_scores, keys)
    print(len(in_scores))

    i = 0
    out_scores = []
    for batch in tqdm(test_ood_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids=centroids, delta=delta)
            #for i, _ in enumerate(ood_keys["softmax"]):
            #    ood_keys["softmax"][i] = get_treshold("sofmtax", ood_keys["softmax"][i])
            out_scores.append(ood_keys)

        if i == 2:
            break
    out_scores = merge_keys(out_scores, keys)
    print(len(out_scores))

    outputs = {}
    for key in keys:


        if key != "softmax":
            continue
        ins = np.array(in_scores[key], dtype=np.float64)
        print(ins)
        outs = np.array(out_scores[key], dtype=np.float64)
        inl = np.ones_like(ins).astype(np.int64)
        outl = np.zeros_like(outs).astype(np.int64)
        scores = np.concatenate([ins, outs], axis=0)
        labels = np.concatenate([inl, outl], axis=0)

        in_acc = (ins == inl).mean().item()
        out_acc = (outs == outl).mean().item()
        print(in_acc)
        print(out_acc)

        auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

        outputs[tag + "_" + key + "_auroc"] = auroc
        outputs[tag + "_" + key + "_fpr95"] = fpr_95

    #wandb.log(outputs) if args.wandb == "log" else None


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


def detect_ood_DNNC(args, model, tokenizer, train, test_id, test_ood):

    def model_predict(data):

        model.eval()

        input = [InputExample(premise, hypothesis) for (premise, hypothesis) in data]

        eval_features = convert_examples_to_features(args, input, tokenizer, train = False)
        input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        max_len = input_mask.sum(dim=1).max().item()
        input_ids = input_ids[:, :max_len]
        input_mask = input_mask[:, :max_len]
        segment_ids = segment_ids[:, :max_len]

        CHUNK = 500
        EXAMPLE_NUM = input_ids.size(0)
        label_list = ["non_entailment", "entailment"]
        labels = []
        probs = None
        start_index = 0

        while start_index < EXAMPLE_NUM:
            end_index = min(start_index+CHUNK, EXAMPLE_NUM)
            
            input_ids_ = input_ids[start_index:end_index, :].to(args.device)
            input_mask_ = input_mask[start_index:end_index, :].to(args.device)
            segment_ids_ = segment_ids[start_index:end_index, :].to(args.device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids_, attention_mask=input_mask_, token_type_ids=segment_ids_)
                logits = outputs[0]
                probs_ = torch.softmax(logits, dim=1)

            probs_ = probs_.detach().cpu()
            if probs is None:
                probs = probs_
            else:
                probs = torch.cat((probs, probs_), dim = 0)
            labels += [label_list[torch.max(probs_[i], dim=0)[1].item()] for i in range(probs_.size(0))]
            start_index = end_index

        assert len(labels) == EXAMPLE_NUM
        assert probs.size(0) == EXAMPLE_NUM
            
        return labels, probs

    def predict_intent(text):

        sampled_train = sample_example(train)

        nli_input = []
        for t in sampled_train:
            for e in t['examples']:
                nli_input.append((text, e)) #Satz, der zu predicten ist, mit allen anderen Trainierten Sätzen kreuzen (text, trained_text)

        assert len(nli_input) > 0

        results = model_predict(nli_input)
        maxScore, maxIndex = results[1][:, 0].max(dim = 0)

        maxScore = maxScore.item()
        maxIndex = maxIndex.item()

        index = -1
        for t in sampled_train:
            for e in t['examples']:
                index += 1

                if index == maxIndex:
                    intent = t['task']
                    matched_example = e

        return intent, maxScore, matched_example

    #for e in tqdm(test_id, desc = 'Intent examples'):
    for e in test_id:
        pred, conf, matched_example = predict_intent(e.text)
        print("-----------------")
        print(e.text)
        print(pred)
        print(conf)
        print(matched_example)
        print("-----------------")
    

