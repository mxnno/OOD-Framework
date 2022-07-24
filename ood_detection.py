from tkinter import X
import torch
import numpy as np
import copy
from copy import deepcopy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
from utils.utils import get_labels, get_num_labels, save_logits
from utils.utils_DNNC import *
from utils.utils_ADB import euclidean_metric
from model_temp import ModelWithTemperature
from scipy.stats import entropy
from scipy.stats import norm as dist_model
from sklearn import svm 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from functools import reduce
from evaluation import evaluate_metriken_ohne_Treshold, evaluate_mit_Treshold, evaluate_scores_ohne_Treshold, evaluate_NLI, evaluate_ADB, evaluate_mit_OOD, evaluate_method_combination
from scipy.stats import chi2
from model import  set_model
import csv
import math


#Paper -> soll mit in die Arbeit


def detect_ood(args, model, train_dataset, train_dev_dataset, dev_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None, best_temp=None):
    

    #OOD Detection + Eval in 3 Schritten
    # 1. Model prediction (BATCH SIZE = 1 !!)
    # ToDo: - Varianz + OC-SVM Score (z.B. nach Mahascore zusätzlich) hinzufügen -> Anhand 1-2 Beispielen testen, ob Mehrwert oder nicht
    # 2. Threshold anwenden -> 0 oder 1
    # **************************
    # * - Tresholdunabhängige Metriken -> wichtig: Maha geht nur in Kombi mit OC-SVM, Softmax nicht
    # * - best Treshold (-> um zu vergleichen, falls man irgendwie auf den Treshold kommen kann) (wie z.B. DNNC oder clinc)
    # * - OC-SVM (alleine nicht so gut wie optimale Treshold )
    # * - DOC -> kein Treshold notwendig
    # * - avg Treshold (über Train/Dev und z.B. über Mean) -> hier nur evtl. gibt gaaanz viele Möglichkeiten (falls noch Zeit Extra Kapitel)
    # *************************
    # WICHTIG: wie komme ich auf T
    #       1. Über Testdaten selber -> besten Treshold picken 
    #       2. Über DEV-Daten (min, mittelwert-Varianz...)
    #       3. Über SVM, log.Regression (geht wsl nicht)
    #       4. DOC anschauen, da braucht man keine Tresholds, UNK -> LOF (https://github.com/thuiar/TEXTOIR/blob/main/open_intent_detection/methods/DeepUnk/manager.py#L146)
    #       5. ID3 Methode (was ist unseen und seen?) mit testID vs TestOOD kommen gleiche ergebnisse wie bei mir raus.. evtl train vs dev oder schauen, dass man tp und fp aus den anderen mehtoden ohne Trehsold bekommt -> aber wsl viel zu hoher Treshold
    # 3. Metriken berechnen
    # - mit T
    # - ohne T (auroc-score (ID1 hier schon vorhanden), tnr_at_tpr95 (ID2), + weitere -> siehe unter tnr_at_tpr95 Evaluation)
    #   WICHTIG: Prüfen ob DTACC mit Treshold .5 * ...



    #Paper
    #Idee: 3 besten Kombinieren (001) -> 0 und (101) -> 1
    #bzw. nicht nur die 3 besten, sondern starke und schwache(d.h. wenn 0.3 in_acc und 0.996 out_rec --> ID sagt, dann sicher ID, wenn nicht dann mit weniger starken weitermachen)

    #Paper
    #Idee: Maha -> SVM -> Treshold: mit optimalem Treshold keine besseren Ergebnisse als ohne SVM(vlt für softmax etc anders? -> nochmal testen)


    #Wichtig:
    #bei Maha muss prepare und Trehsold predriction auf unterschiedlichen Daten ausgeführt werden, d.h. z.B. prepare mit Training und Treshold mit dev -> sonst kommen immer die gleiche Werte raus (15 oder 90 für alle Maha Predictions)
    model.prepare_ood(train_dataset)


    ################# Logit, Pool und Label für TRAIN|DEV|IN|OUT 
    #Train 
    train_labels = []
    for batch in tqdm(train_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        train_label = batch["labels"].cpu().detach()
        with torch.no_grad():
            model.compute_ood_outputs(**batch)
        train_labels.append(train_label)
        
    all_logits_train = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_train = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])
    train_labels = reduce(lambda x,y: torch.cat((x,y)), train_labels[::])

    #zum Abspeichern der logits und pools
    #save_logits(all_logits_train, 'all_logits_train.pt')
    #save_logits(all_pool_train, '/content/drive/MyDrive/Masterarbeit/Results/1305_train_pool.pt')
    model.all_logits = []
    model.all_pool = []

    #Dev:
    dev_labels = []
    for batch in tqdm(dev_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        dev_label = batch["labels"].cpu().detach()
        with torch.no_grad():
            model.compute_ood_outputs(**batch)
        dev_labels.append(dev_label)
    
    all_logits_dev = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_dev = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])
    dev_labels = reduce(lambda x,y: torch.cat((x,y)), dev_labels[::])
    #save_logits(all_logits_out, 'all_ood_logits.pt')
    #save_logits(all_pool_dev, '/content/drive/MyDrive/Masterarbeit/Results/1305_dev_treshold_pool.pt')
    model.all_logits = []
    model.all_pool = []


    #Test-ID:
    for batch in tqdm(test_id_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            model.compute_ood_outputs(**batch)
        
    all_logits_in = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_in = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])


    #zum Abspeichern der logits und pools
    #save_logits(all_logits_in, '1305_traindev_id_logits.pt')
    #save_logits(all_pool_in, '/content/drive/MyDrive/Masterarbeit/Results/1305_dev_id_pool.pt')
    model.all_logits = []
    model.all_pool = []
    
    #Test-OOD:
    for batch in tqdm(test_ood_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            model.compute_ood_outputs(**batch)
           
    all_logits_out = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_out = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])
    #zum Abspeichern der logits und pools
    #save_logits(all_logits_out, '1305_traindev_ood_logits.pt')
    #save_logits(all_pool_out, '/content/drive/MyDrive/Masterarbeit/Results/1305_dev_ood_pool.pt')
    model.all_logits = []
    model.all_pool = []


    if args.model_ID == 14:
        n, i, o = detect_ood_adb(args, centroids, delta, all_pool_in, all_pool_out)
        return n,i,o
        #all_logits_train, all_logits_dev, all_logits_in , all_logits_out = detect_ood_adb2(centroids, delta, all_pool_in, all_pool_out, all_pool_dev, all_pool_train)


#2. Treshold + Scores
    
    #scores_dev = Scores(thresholds, all_logits_dev, all_logits_out, all_pool_dev, all_pool_out, all_logits_train, all_pool_train, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    #scores_dev.calculate_scores(best_temp)
    #

    thresholds = Tresholds()
    scores = Scores(all_logits_in, all_logits_out, all_pool_in, all_pool_out, all_logits_train, all_pool_train, all_logits_dev, all_pool_dev, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    print("Calculate all scores...")
    scores.calculate_scores(best_temp, args)


    
    
    # (muss als erstes)
# 2.1 Metriken ohne Treshold -> man braucht die Scores, nicht 0,1 ...
    # -> nicht für LOF und DOC möglich
    # -> Maha nur in Kombi mit OC-SVM-Scores
    # --> bei softmax wesentlich schlechtere Erg mit OC-SVM -> softmax mit und ohne OC-SVM Scores in einer Tabelle
    print("Metriken ohne Treshold...")
    scores_ocsvm = deepcopy(scores)
    scores_ocsvm.apply_ocsvm(args, "scores")
    #-> scores_in bzw. scors_out UND scores_in_ocsvm bzw. scores_out_ocsvm in eval abfragen!
    #evaluate_metriken_ohne_Treshold(args, scores_ocsvm)



# 2.2 mit Treshold (best. avg)

    print("Mit Treshold...")
    print("...best...")
    scores_best = deepcopy(scores)
    thresholds.calculate_tresholds(args, scores_best, 'best')
    scores_best_copy = deepcopy(scores_best)
    scores_best.apply_tresholds(args, thresholds, all_pool_in, all_pool_out, centroids, delta)
    n, i, o = scores_best.calculate_method_combination(args, scores_best_copy, thresholds, sType="alle_methoden")

    return n, i, o
    evaluate_method_combination(args, n, i, o, "best")
    #evaluate_mit_Treshold(args, scores_best, 'best')
    print("...best_dev...")
    scores_best_dev = deepcopy(scores)
    thresholds.calculate_tresholds(args, scores_best_dev, 'best_dev')
    scores_best_dev.apply_tresholds(args, thresholds, all_pool_in, all_pool_out, centroids, delta)
    #evaluate_mit_Treshold(args, scores_best_dev, 'best_dev')
    # print("...avg...")
    # scores_avg = deepcopy(scores)
    # thresholds.calculate_tresholds(args, scores_avg, 'avg')
    # scores_avg_copy = deepcopy(scores_avg)
    # scores_avg.apply_tresholds(args, thresholds, all_pool_in, all_pool_out, centroids, delta)
    # n, i, o = scores_avg.calculate_method_combination(args, scores_avg_copy, thresholds, sType="alle_methoden")
    # evaluate_method_combination(args, n, i, o, "avg")
    #evaluate_mit_Treshold(args, scores_avg, 'avg')
    
# 2.2 ohne Treshold zu 0/1
    # - OCSVM Predict (logits, softmax ...)
    # - DOC (logits) -> geht nur für alleine
    # - LOF (logits+pool) -> geht nur für alleine
    print("Scores ohne Treshold...")
    #für ADB 
    
    scores_ohne_Treshold = deepcopy(scores)
    scores_ohne_Treshold.apply_ocsvm(args, 'predict')
    #evaluate_scores_ohne_Treshold(args, scores_ohne_Treshold)


   
def detect_ood_adb(args, centroids, delta, pool_in, pool_out):

    
    def get_adb_score(pooled, centroids, delta):

        logits_adb = euclidean_metric(pooled, centroids)
        #qwert
        #Kann ich die logits rausnehmen für anderen Methoden???
        # -> heir ja nur Softmax
        #probs, preds = F.softmax(logits_adb.detach(), dim = 1).max(dim = 1)
        
        probs, preds = logits_adb.max(dim = 1)
        preds_ones = torch.ones_like(preds)
        preds.to('cuda:0')
        euc_dis = torch.norm(pooled - centroids[preds], 2, 1).view(-1)           
        preds_ones[euc_dis >= delta[preds]] = 0
        return preds_ones.detach().cpu().numpy()

    adb_pred_in = get_adb_score(pool_in, centroids, delta)
    adb_pred_out = get_adb_score(pool_out, centroids, delta)

    return ["adb",], [adb_pred_in,], [adb_pred_out,]
   # evaluate_ADB(args, adb_pred_in, adb_pred_out)

def detect_ood_adb2(centroids, delta, pool_in, pool_out, pool_dev, pool_train):

    logits_train = euclidean_metric(pool_train, centroids)
    logits_dev = euclidean_metric(pool_dev, centroids)
    logits_in = euclidean_metric(pool_in, centroids)
    logits_out = euclidean_metric(pool_out, centroids)

    return logits_train, logits_dev, logits_in, logits_out



def detect_ood_DNNC(args, model, tokenizer, train, test_id, test_ood):

    def model_predict(data, method="logits"):

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
        #label_list = ["non_entailment", "entailment"]
        label_list = ["entailment", "non_entailment"]
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


                if method == "softmax":
                    probs_ = torch.softmax(logits, dim=1)
                elif method == "logits":
                    probs_ = logits

            probs_ = probs_.detach().cpu()
            if probs is None:
                probs = probs_
            else:
                probs = torch.cat((probs, probs_), dim = 0)

                #50/50 Entscheidung ob non_etailment oder entailemnt
                # was anderes als Softmax möglich?
            labels += [label_list[torch.max(probs_[i], dim=0)[1].item()] for i in range(probs_.size(0))]
            start_index = end_index

        assert len(labels) == EXAMPLE_NUM
        assert probs.size(0) == EXAMPLE_NUM
        
        # return labgel liste für das eine Beispiel kombiniert mit allen Trainingsdaten ['non_entailment', 'entailment', 'non_entailment'...]
        # return probs für die Labels für die beiden Klassen: [[0.9804, 0.0196],[0.9804, 0.0196],...]
        return labels, probs

    def predict_intent(text):

        sampled_train = sample_example(train)

        nli_input = []
        for t in sampled_train:
            for e in t['examples']:
                nli_input.append((text, e)) #Satz, der zu predicten ist, mit allen anderen Trainierten Sätzen kreuzen (text, trained_text)

        assert len(nli_input) > 0

        results = model_predict(nli_input)
        #results[1] = [[0.9804, 0.0196],[0.9804, 0.0196],...] -> linke Seite wsl für non_ent., rechts entailment
        #-> maxScore, maxIndex = results[1][:, 1].max(dim = 0) # -> non entail max
        maxScore, maxIndex = results[1][:, 0].max(dim = 0) # -> entail max

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

    #for e in tqdm(test_id, desc = 'Intent examples')
    pred_id = []
    pred_ood = []

    for e in tqdm(test_id, desc = 'ID examples'):
        pred, conf, matched_example = predict_intent(e.text)

        pred_id.append(conf)
        # print("-----------------")
        # print(e.text)
        # print(e.label)
        # print(pred)
        # print(conf)
        # print(matched_example)
        # print("-----------------")

    for e in tqdm(test_ood, desc = 'OOD examples'):
        pred, conf, matched_example = predict_intent(e.text)
        pred_ood.append(conf)

        # print("-----------------")
        # print(e.text)
        # print(e.label)
        # print(pred)
        # print(conf)
        # print(matched_example)
        # print("-----------------")

    
    pred_id = np.array(pred_id)
    pred_id = np.where(pred_id >= 0.5, 1, 0)

    pred_ood = np.array(pred_ood)
    pred_ood = np.where(pred_ood >= 0.5, 1, 0)

    return ["nli",], [pred_id,], [pred_ood,]
    #evaluate_NLI(args, pred_id, pred_ood)










#################################################### METHODEN ####################################################



def get_model_prediction(logits, full_score=False, idx=False):
    #return label_pred des Models
    if full_score:
        return logits

    pred, label_pred = logits.max(dim = 1)

    if idx:
        return label_pred.cpu().detach().numpy()
    else:
        return pred.cpu().detach().numpy()


def get_varianz_score(logits):

    var_score = logits.cpu().detach().numpy()
    return np.var(var_score, axis=1)

def get_softmax_score(logits, full_score=False):
    # Nach ID 2 (=Maxprob)
    # Input: logits eines batches
    # Return :
    # - softmax_score(Scores der prediction für alle Klassen -> max(-1) gibt den Max Wert aus)
    # - max_indices = Klasse, die den Max Wert hat 

    if full_score:

        softmax = F.softmax(logits, dim=-1)
        return softmax

    softmax, idx = F.softmax(logits, dim=-1).max(-1)
    return softmax.cpu().detach().numpy()


def get_softmax_score_with_temp(logits, temp, full_score=False, idx=False):
    # Nach Odin: https://openreview.net/pdf?id=H1VGkIxRZ
    # Input: 
    # - logits eines batches
    # - temp (low temp (below 1) -> makes the model more confident, high temperature (above 1) makes the model less confident)

    # Return :
    # - softmax_score(Scores der prediction für alle Klassen -> max(-1) gibt den Max Wert aus)
    # - max_indices = Klasse, die den Max Wert hat

    if full_score:

        softmax = F.softmax((logits / temp), dim=-1)
        return softmax

    softmax, softmax_idx = F.softmax((logits / temp), dim=-1).max(-1)

    if idx:
        return idx.cpu().detach().numpy()
    else:
        return softmax.cpu().detach().numpy()

def get_entropy_score(logits):

    #-> hier kein Full_score möglich -> nicht für OCSVM geeignet...

    #Nach ID 11 (Gold)
    # Input: 
    # - logits eines batches
    # Return :
    # - entropy score

    softmax = F.softmax(logits, dim=-1)
    expo = torch.exp(softmax)
    expo = expo.cpu()
    return entropy(expo, axis=1)

def get_cosine_score(pooled, norm_bank, full_scores=False, idx=False):
    #von ID 2
    #hier fehlt noch norm_bank
    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = norm_pooled @ norm_bank.t()
    if full_scores is True:
        return cosine_score
    
    if idx:
        idx = cosine_score.max(-1)[1]
        return idx.cpu().detach().numpy()
    else:
        cosine_score = cosine_score.max(-1)[0]
        return cosine_score.cpu().detach().numpy()


def get_energy_score(logits):
    #-> hier kein Full_score möglich -> nicht für OCSVM geeignet...

    #von ID 2
    return torch.logsumexp(logits, dim=-1).cpu().detach().numpy()


def get_maha_score(pooled, all_classes, class_mean, class_var, full_scores=False, idx=False):

    #ID 2:
    maha_score = []
    for c in all_classes:
        centered_pooled = pooled - class_mean[c].unsqueeze(0)
        ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
        maha_score.append(ms)
    maha_score_full = torch.stack(maha_score, dim=-1)
    if full_scores is True:
        return maha_score_full

    if idx:
        idx = maha_score_full.min(-1)[1]
        return idx.cpu().detach().numpy()
    else:
        maha_score = maha_score_full.min(-1)[0]
        return maha_score.cpu().detach().numpy()
    #https://www.statology.org/mahalanobis-distance-python/
    # aus distanz ein p-Value berechnen -> outlier



def get_gda_score(train_logits, test_logits_in, test_logits_out, dev_logits, train_labels, distance_type, full_scores):

    #andere Methode um Means und Varianz zu bekommen -> ansonsten genauso wie maha

    def gda_help(prob_test, means, distance_type, cov):
        num_samples = prob_test.shape[0]
        num_features = prob_test.shape[1]
        num_classes = means.shape[0]
        if distance_type == "euclidean":
            cov = np.identity(num_features)
        features = prob_test.reshape(num_samples, 1, num_features).repeat(num_classes,axis=1)  # (num_samples, num_classes, num_features)
        means = means.reshape(1, num_classes, num_features).repeat(num_samples, axis=0)  # (num_samples, num_classes, num_features)
        vectors = features - means  # (num_samples, num_classes, num_features)
        cov_inv = np.linalg.inv(cov)
        bef_sqrt = np.matmul(np.matmul(vectors.reshape(num_samples, num_classes, 1, num_features), cov_inv),
                              vectors.reshape(num_samples, num_classes, num_features, 1)).squeeze()
        result = np.sqrt(bef_sqrt)
        result[np.isnan(result)] = 1e12  # solve nan
        if full_scores:
            return result
        result = result.min(axis=1)

        return result

  #= Maha oder Eucliien mit anderer Methode
  #ID 3+4 machen es so mit LinearDiscriminantAnalysis: https://github.com/pris-nlp/Generative_distance-based_OOD/blob/main/experiment.py#L248

    prob_train = F.softmax(train_logits, dim=-1)
    prob_train = prob_train.cpu().detach().numpy()
    prob_dev = F.softmax(dev_logits, dim=-1)
    prob_dev = prob_dev.cpu().detach().numpy()

    prob_test_in = F.softmax(test_logits_in, dim=-1)
    prob_test_in = prob_test_in.cpu().detach().numpy()
    prob_test_out = F.softmax(test_logits_out, dim=-1)
    prob_test_out = prob_test_out.cpu().detach().numpy()

    
    #solver {‘svd’, ‘lsqr’, ‘eigen’}
    gda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, store_covariance=True)
    train_labels.cpu().detach().numpy()
    gda.fit(prob_train, train_labels)

    means =  gda.means_
    cov = gda.covariance_

    results_train = gda_help(prob_train, means, distance_type, cov)
    results_dev = gda_help(prob_dev, means, distance_type, cov)
    results_in = gda_help(prob_test_in, means, distance_type, cov)
    results_out = gda_help(prob_test_out, means, distance_type, cov)
    
    return results_train, results_dev, results_in, results_out


def get_lof_score(logits_in, logits_out, pooled_in, pooled_out, train_pooled, args=None):
    #Falls es OOD_Trainingsdaten gibt -> Outlier, sosnt Novelety -> novelty=True setzen
    # Logits sind nicht batchweise sonder alle !
    #aus: https://github.com/thuiar/TEXTOIR/blob/main/open_intent_detection/methods/DeepUnk/manager.py
    #Org paper: https://arxiv.org/pdf/1906.00434.pdf

    #https://github.com/pris-nlp/Generative_distance-based_OOD/blob/main/experiment.py#L227
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

    #ein großer Tensor mit allen Train Pooling Daten
    train_feats = train_pooled.cpu().numpy()


    #wir brauchen aber die predicteten label des Models
    _, pred_labels_in = logits_in.max(dim = 1)
    _, pred_labels_out = logits_out.max(dim = 1)
    pred_labels_in = pred_labels_in.cpu().numpy()
    pred_labels_out = pred_labels_out.cpu().numpy()
    #ein großer Tensor für Pooling 
    pooled_in = pooled_in.cpu().numpy()
    pooled_out = pooled_out.cpu().numpy()


    #lof = LocalOutlierFactor(n_neighbors=20, metric = "mahalanobis", metric_params={'VI': np.cov(train_feats, rowvar= False)},   novelty=True, n_jobs=-1)
    lof = LocalOutlierFactor(n_neighbors=args.few_shot, novelty=True, contamination = 0.5, n_jobs=-1)
    lof.fit(train_feats)
    #predict() -> -1 OOD, 1 ID
    y_pred_lof_in = lof.predict(pooled_in)
    pred_labels_in = np.ones_like(y_pred_lof_in)
    pred_labels_in[y_pred_lof_in == -1] = 0

    #pooled out: [1000*768] 1000 wegen 1000 Testdaten
    y_pred_lof_out = lof.predict(pooled_out)
    pred_labels_out = np.ones_like(y_pred_lof_out)
    pred_labels_out[y_pred_lof_out == -1] = 0

    # Orginal: geben preds (pred_labels) zurück und setzen das Label der OOD auf data.unseen_label_id
    #preds[y_pred_lof[y_pred_lof == -1].index] = data.unseen_label_id
    return pred_labels_in, pred_labels_out
  



def get_doc_score(train_logits, train_labels, logits_predict, all_classes):
    #bzw DOC ID 12/13

    if not isinstance(train_logits, np.ndarray):
        train_logits = train_logits.cpu().detach().numpy()
    if not isinstance(logits_predict, np.ndarray):
        logits_predict = logits_predict.cpu().detach().numpy()

    def mu_fit(prob_pos_X):
        prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std

    mu_stds = []
    for i in range(len(all_classes)):
        index_list = [y for y, x in enumerate(train_labels) if x == i]
        pos_mu, pos_std = mu_fit(train_logits[index_list, i])
        mu_stds.append([pos_mu, pos_std])
    
    thresholds = np.empty([len(all_classes)])
    for col in range(len(all_classes)):
        threshold = mu_stds[col][1]
        label = all_classes[col]
        thresholds[label] = threshold
    thresholds = np.array(thresholds)
    #treshold = np.mean(thresholds) + np.var(thresholds)
    threshold = np.max(thresholds)
    threshold = np.mean(thresholds)

    y_pred = []
    for p in logits_predict:
        max_class = np.argmax(p)
        max_value = np.max(p)
        
        #threshold = max(0.5, 1 - 0.3 * mu_stds[max_class][1])
        #threshold = mu_stds[max_class][1]
        if max_value > threshold:
            #y_pred.append(max_class)
            y_pred.append(1)
        else:
            #y_pred.append(data.unseen_label_id)
            y_pred.append(0)

    return np.array(y_pred)



#Reihenfolge




class Scores():

    def __init__(self, logits_in, logits_out, pooled_in, pooled_out, logits_train, pooled_train, logits_dev, pooled_dev, norm_bank, all_classes, train_labels, dev_labels, class_mean, class_var):

        #Logits & Pooled
        self.logits_train = logits_train
        self.pooled_train = pooled_train

        self.logits_dev = logits_dev
        self.pooled_dev = pooled_dev

        self.logits_in = logits_in
        self.logits_out = logits_out
        self.pooled_in = pooled_in
        self.pooled_out = pooled_out
        
        #sonstiges
        
        self.norm_bank = norm_bank
        self.all_classes = all_classes # all_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.train_labels = train_labels # labels aller Trainingsdaten: [1, 11, 3, 1, 14, ...]
        self.dev_labels = dev_labels
        self.class_mean = class_mean
        self.class_var = class_var

        #Scores

        self.logits_score_in_ocsvm = 0
        self.logits_score_out_ocsvm = 0
        self.logits_score_train_ocsvm = 0
        self.logits_score_train = 0
        self.logits_score_dev = 0
        self.logits_score_in = 0
        self.logits_score_out = 0
        self.logits_score_full = 0

        self.varianz_score_train = 0
        self.varianz_score_dev = 0
        self.varianz_score_in = 0
        self.varianz_score_out = 0
        self.varianz_score_full = 0
        
        self.softmax_score_in_ocsvm = 0
        self.softmax_score_out_ocsvm = 0
        self.softmax_score_train_ocsvm = 0
        self.softmax_score_train = 0
        self.softmax_score_dev = 0
        self.softmax_score_in = 0
        self.softmax_score_out = 0
        self.softmax_score_full = 0

        self.softmax_score_temp_in_ocsvm = 0
        self.softmax_score_temp_out_ocsvm = 0
        self.softmax_score_temp_train_ocsvm = 0
        self.softmax_score_temp_train = 0
        self.softmax_score_temp_dev = 0
        self.softmax_score_temp_in = 0
        self.softmax_score_temp_out = 0
        self.softmax_score_temp_full = 0

        self.cosine_score_in_ocsvm = 0
        self.cosine_score_out_ocsvm = 0
        self.cosine_score_train_ocsvm = 0
        self.cosine_score_train = 0
        self.cosine_score_dev = 0
        self.cosine_score_in = 0
        self.cosine_score_out = 0
        self.cosine_score_full = 0

        self.energy_score_train = 0
        self.energy_score_dev = 0
        self.energy_score_in = 0
        self.energy_score_out = 0
        self.energy_score_full = 0

        self.entropy_score_train = 0
        self.entropy_score_dev = 0
        self.entropy_score_in = 0
        self.entropy_score_out = 0
        self.entropy_score_full = 0

        #GDA ist wie Softmax, Maha etc.
        self.gda_maha_score_dev_ocsvm = 0
        self.gda_maha_score_in_ocsvm = 0
        self.gda_maha_score_out_ocsvm = 0
        self.gda_maha_score_train = 0
        self.gda_maha_score_dev = 0
        self.gda_maha_score_in = 0
        self.gda_maha_score_out = 0
        self.gda_maha_score_full = 0

        self.gda_eucl_score_dev_ocsvm = 0
        self.gda_eucl_score_in_ocsvm = 0
        self.gda_eucl_score_out_ocsvm = 0
        self.gda_eucl_score_train = 0
        self.gda_eucl_score_dev = 0
        self.gda_eucl_score_in = 0
        self.gda_eucl_score_out = 0
        self.gda_eucl_score_full = 0

        self.maha_score_in_ocsvm = 0
        self.maha_score_out_ocsvm = 0
        self.maha_score_dev_ocsvm = 0
        self.maha_score_train = 0
        self.maha_score_dev = 0
        self.maha_score_in = 0
        self.maha_score_out = 0
        self.maha_score_full = 0

        self.doc_score_in = 0
        self.doc_score_out = 0
        self.doc_score_full = 0
        
        self.lof_score_in = 0
        self.lof_score_out = 0



        #Kombinationen:
        self.varianz_cosine_energy_in = 0
        self.varianz_cosine_energy_out = 0

        self.varianz_energy_doc_in = 0
        self.varianz_energy_doc_out = 0

        self.varianz_energy_maha_in = 0
        self.varianz_energy_maha_out = 0

        self.softmax_temp_cosine_maha_in = 0
        self.softmax_temp_cosine_maha_in = 0


    def calculate_scores(self, best_temp, args=None):


        ################## Nur Logits #####################
        self.logits_score_in_ocsvm = get_model_prediction(self.logits_in, True)
        self.logits_score_out_ocsvm = get_model_prediction(self.logits_out, True)
        self.logits_score_train_ocsvm = get_model_prediction(self.logits_train, True)
        self.logits_score_train = get_model_prediction(self.logits_train, False)
        self.logits_score_dev = get_model_prediction(self.logits_dev, False)
        self.logits_score_in = get_model_prediction(self.logits_in, False)
        self.logits_score_out = get_model_prediction(self.logits_out, False)

        ################## VARIANZ #####################
        self.varianz_score_train = get_varianz_score(self.logits_train)
        self.varianz_score_dev = get_varianz_score(self.logits_dev)
        self.varianz_score_in = get_varianz_score(self.logits_in)
        self.varianz_score_out = get_varianz_score(self.logits_out)

        ################## SOFTMAX ########################
        self.softmax_score_in_ocsvm = get_softmax_score(self.logits_in, True)
        self.softmax_score_out_ocsvm = get_softmax_score(self.logits_out, True)
        self.softmax_score_train_ocsvm = get_softmax_score(self.logits_train, True)
        self.softmax_score_train = get_softmax_score(self.logits_train, False)
        self.softmax_score_dev = get_softmax_score(self.logits_dev, False)
        self.softmax_score_in = get_softmax_score(self.logits_in, False)
        self.softmax_score_out= get_softmax_score(self.logits_out, False)

        ################# SOFTMAX TMP #####################
        self.softmax_score_temp_in_ocsvm = get_softmax_score_with_temp(self.logits_in, best_temp, full_score=True)
        self.softmax_score_temp_out_ocsvm = get_softmax_score_with_temp(self.logits_out, best_temp, full_score=True)
        self.softmax_score_temp_train_ocsvm = get_softmax_score_with_temp(self.logits_train, best_temp, full_score=True)
        self.softmax_score_temp_train = get_softmax_score_with_temp(self.logits_train, best_temp, full_score=False)
        self.softmax_score_temp_dev = get_softmax_score_with_temp(self.logits_dev, best_temp, full_score=False)
        self.softmax_score_temp_in = get_softmax_score_with_temp(self.logits_in, best_temp, full_score=False)
        self.softmax_score_temp_out = get_softmax_score_with_temp(self.logits_out, best_temp, full_score=False)

        ################# COSINE ##########################
        self.cosine_score_in_ocsvm = get_cosine_score(self.pooled_in, self.norm_bank, True)
        self.cosine_score_out_ocsvm = get_cosine_score(self.pooled_out, self.norm_bank, True)
        self.cosine_score_dev_ocsvm = get_cosine_score(self.pooled_dev, self.norm_bank, True)
        self.cosine_score_train = get_cosine_score(self.pooled_train, self.norm_bank, False)
        self.cosine_score_dev = get_cosine_score(self.pooled_dev, self.norm_bank, False)
        self.cosine_score_in = get_cosine_score(self.pooled_in, self.norm_bank, False)
        self.cosine_score_out = get_cosine_score(self.pooled_out, self.norm_bank, False)

        ################### ENERGY #######################
        self.energy_score_train = get_energy_score(self.logits_train)
        self.energy_score_dev = get_energy_score(self.logits_dev)
        self.energy_score_in = get_energy_score(self.logits_in)
        self.energy_score_out = get_energy_score(self.logits_out)

        ################## ENTROPY ########################
        self.entropy_score_train = get_entropy_score(self.logits_train)
        self.entropy_score_dev = get_entropy_score(self.logits_dev)
        self.entropy_score_in = get_entropy_score(self.logits_in)
        self.entropy_score_out = get_entropy_score(self.logits_out)

        ################### GDA ############################
        # ist maha und euclid
        if args.model_ID != 14:
            self.gda_maha_score_train_ocsvm, self.gda_maha_score_dev_ocsvm, self.gda_maha_score_in_ocsvm , self.gda_maha_score_out_ocsvm = get_gda_score(self.logits_train, self.logits_in, self.logits_out, self.logits_dev, self.train_labels, "maha", True)
            self.gda_maha_score_train, self.gda_maha_score_dev, self.gda_maha_score_in , self.gda_maha_score_out = get_gda_score(self.logits_train, self.logits_in, self.logits_out, self.logits_dev, self.train_labels, "maha", False)
            
            self.gda_eucl_score_train_ocsvm, self.gda_eucl_score_dev_ocsvm, self.gda_eucl_score_in_ocsvm , self.gda_eucl_score_out_ocsvm = get_gda_score(self.logits_train, self.logits_in, self.logits_out, self.logits_dev, self.train_labels, "euclidean", True)
            self.gda_eucl_score_train, self.gda_eucl_score_dev, self.gda_eucl_score_in , self.gda_eucl_score_out = get_gda_score(self.logits_train, self.logits_in, self.logits_out, self.logits_dev, self.train_labels, "euclidean", False)
        
        ################### MAHA ############################
        self.maha_score_in_ocsvm = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_out_ocsvm = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_dev_ocsvm = get_maha_score(self.pooled_dev, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_train_ocsvm = get_maha_score(self.pooled_train, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_train = get_maha_score(self.pooled_train, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_dev = get_maha_score(self.pooled_dev, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_in = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_out = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, False)


        #################### LOF ############################
        #(return 0/1 -> kein treshold notwendig)
        self.lof_score_in, self.lof_score_out = get_lof_score(self.logits_in, self.logits_out, self.pooled_in, self.pooled_out, self.pooled_train, args)

        #################### DOC ##########################
        #(return 0/1 -> kein treshold notwendig)
        self.doc_score_in = get_doc_score(self.logits_train, self.train_labels, self.logits_in, self.all_classes)
        self.doc_score_out = get_doc_score(self.logits_train, self.train_labels, self.logits_out, self.all_classes)


    def calculate_method_combination(self, args, score_copy, tresholds, sType="abc"):

        #idee: alle und höchste confidenc wählt aus
        list_in = []
        list_out = []
        list_kombi = []

        #idee2: mehrheit 

        methods = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha", "doc"]
        for m in methods:
            if m == "softmax_temp":
                    namex = "softmax_score_temp_"
            else:
                namex = m + "_score_"

            m_in = getattr(self,  namex + "in")
            m_out = getattr(self,  namex + "out")
            list_in.append(m_in)
            list_out.append(m_out)
            list_kombi.append(m)



        #weighted mehrheit
        # sum(x*1 bzw. x*-1) --> > 0 = 1, < = 0

        methods = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha"]
        
        score_in = np.empty_like(self.varianz_score_in)
        
        for i, _ in enumerate(self.varianz_score_in):

            diff = 0
            for x in methods:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = getattr(score_copy,  namex + "in")

                t1 = getattr(tresholds, x + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)


                weight = math.exp(m_score)
        
                best_in = getattr(self,  namex + "in")

                if best_in[i] == 0:
                    diff += weight * -1
                else:
                    diff += weight * 1

            if diff >= 0:
                score_in[i] = 1
            else:
                score_in[i] = 0

        score_out = np.empty_like(self.varianz_score_out)

        for i, _ in enumerate(self.varianz_score_out):

            diff = 0
            for x in methods:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = getattr(score_copy,  namex + "out")

                t1 = getattr(tresholds, x + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)

                
                weight = math.exp(min(m_score, 20))
        
                best_out = getattr(self,  namex + "out")

                if best_out[i] == 0:
                    diff += weight * -1
                else:
                    diff += weight * 1


            if diff >= 0:
                score_out[i] = 1
            else:
                score_out[i] = 0
    

        list_in.append(score_in)
        list_out.append(score_out)
        list_kombi.append("weighted_mehrheit")


        # mean confience 1 vs mean confidence 0
        methods = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha"]
        
        score_in = np.empty_like(self.varianz_score_in)
        
        for i, _ in enumerate(self.varianz_score_in):
            counter0 = 0
            counter1 = 0

            sum1 = 0
            sum0 = 0
            for x in methods:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = getattr(score_copy,  namex + "in")

                t1 = getattr(tresholds, x + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)

                best_in = getattr(self,  namex + "in")

                if best_in[i] == 0:
                    counter0 += 1
                    sum0 += m_score
                else:
                    counter1 += 1
                    sum1 += m_score

            if counter1 > 0:
                mean1 = sum1/counter1
            else: 
                mean1 = 0
            if counter0 > 0:
                mean0 = sum0/counter0
            else:
                mean0 = 0

            if mean1 > mean0:
                score_in[i] = 1
            else:
                score_in[i] = 0

        
        
        score_out = np.empty_like(self.varianz_score_out)
        for i, _ in enumerate(self.varianz_score_out):
            counter0 = 0
            counter1 = 0

            sum1 = 0
            sum0 = 0
            for x in methods:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = getattr(score_copy,  namex + "out")

                t1 = getattr(tresholds, x + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)

                best_out = getattr(self,  namex + "out")

                if best_out[i] == 0:
                    counter0 += 1
                    sum0 += m_score
                else:
                    counter1 += 1
                    sum1 += m_score

            if counter1 > 0:
                mean1 = sum1/counter1
            else: 
                mean1 = 0
            if counter0 > 0:
                mean0 = sum0/counter0
            else:
                mean0 = 0

            if mean1 > mean0:
                score_out[i] = 1
            else:
                score_out[i] = 0

        
        list_in.append(score_in)
        list_out.append(score_out)
        list_kombi.append("mean_confidence")



        
        # # alle höchste confidence
        methods = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha"]
        
        score_in = np.empty_like(self.varianz_score_in)
        
        for i, _ in enumerate(self.varianz_score_in):
            max_diff = 0
            for x in methods:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = getattr(score_copy,  namex + "in")

                t1 = getattr(tresholds, x + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)
                if m_score > max_diff:
                    max_diff = m_score
                    best_in = getattr(self,  namex + "in")

            score_in[i] = best_in[i]
        
        score_out = np.empty_like(self.varianz_score_out)
        
        for i, _ in enumerate(self.varianz_score_out):
            max_diff = 0
            for x in methods:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = getattr(score_copy,  namex + "out")

                t1 = getattr(tresholds, x + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)
                if m_score > max_diff:
                    max_diff = m_score
                    best_out = getattr(self,  namex + "out")

            score_out[i] = best_out[i]


        list_in.append(score_in)
        list_out.append(score_out)
        list_kombi.append("best_confidence")


        # # Top 3 Confdence
        methods = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha"]
        
        score_in = np.empty_like(self.varianz_score_in)
        
        for i, _ in enumerate(self.varianz_score_in):
            max_diff_1 = 0
            m1_v = 0
            max_diff_2 = 0
            m2_v = 0
            max_diff_3 = 0
            m3_v = 0
            for method in methods:
                
                if method == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = method + "_score_"
                m_score = getattr(score_copy,  namex + "in")

                t1 = getattr(tresholds, method + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)
                if m_score > max_diff_2:

                    max_diff_3 = max_diff_2
                    max_diff_2 = max_diff_1
                    max_diff_1 = m_score

                    m3_v = m2_v
                    m2_v = m1_v
                    v_in = getattr(self,  namex + "in")
                    m1_v = v_in[i]

                elif m_score > max_diff_2:
                    
                    max_diff_3 = max_diff_2
                    max_diff_2 = m_score

                    m3_v = m2_v
                    v_in = getattr(self,  namex + "in")
                    m2_v = v_in[i]

                elif m_score > max_diff_3:
                    
                    max_diff_3 = m_score

                    v_in = getattr(self,  namex + "in")
                    m3_v = v_in[i]

            
            m_sum = m1_v + m2_v + m3_v

            if m_sum >= 2:
                score_in[i] = 1
            else:
                score_in[i] = 0


        
        score_out = np.empty_like(self.varianz_score_out)
        
        for i, _ in enumerate(self.varianz_score_out):
            max_diff_1 = 0
            m1_v = 0
            max_diff_2 = 0
            m2_v = 0
            max_diff_3 = 0
            m3_v = 0
            for method in methods:
                
                if method == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = method + "_score_"
                m_score = getattr(score_copy,  namex + "out")

                t1 = getattr(tresholds, method + "_t")
                
                #normalisieren
                m_score = 1/t1 * m_score[i]
                m_score = abs(1-m_score)
                if m_score > max_diff_2:

                    max_diff_3 = max_diff_2
                    max_diff_2 = max_diff_1
                    max_diff_1 = m_score

                    m3_v = m2_v
                    m2_v = m1_v
                    v_out = getattr(self,  namex + "out")
                    m1_v = v_out[i]

                elif m_score > max_diff_2:
                    
                    max_diff_3 = max_diff_2
                    max_diff_2 = m_score

                    m3_v = m2_v
                    v_out = getattr(self,  namex + "out")
                    m2_v = v_out[i]

                elif m_score > max_diff_3:
                    
                    max_diff_3 = m_score

                    v_out = getattr(self,  namex + "out")
                    m3_v = v_out[i]

            
            m_sum = m1_v + m2_v + m3_v

            if m_sum >= 2:
                score_out[i] = 1
            else:
                score_out[i] = 0


        list_in.append(score_in)
        list_out.append(score_out)
        list_kombi.append("top 3 confidence")

        

        #Mehrheit
        print("###############################")
        print("Mehrheit")
        methods = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha", "doc"]
        
        score_in = np.empty_like(self.varianz_score_in)
    
        for i, _ in enumerate(self.varianz_score_in):
            m_sum = 0
            for x in methods:
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = None
                m_score = getattr(self,  namex + "in")
                m_sum += m_score[i]

            if m_sum >= 6:
                score_in[i] = 1
            else:
                score_in[i] = 0


        score_out= np.empty_like(self.varianz_score_out)
        for i, _ in enumerate(self.varianz_score_out):
            m_sum = 0
            for x in methods:
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                m_score = None
                m_score = getattr(self,  namex + "out")
                m_sum += m_score[i]

            if m_sum >=  6:
                score_out[i] = 1
            else:
                score_out[i] = 0         
        

        
        list_in.append(score_in)
        list_out.append(score_out)
        list_kombi.append("Mehrheit 1/0")





        #2 und bei unentschieden höchste confidence

        methods1 = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha"]
        methods2 = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha"]

        hlist_kombi = []
        hlist_in = []
        hlist_out = []

        for i, x in enumerate(methods1):
    
            methods2.pop(0)
            for y in methods2:
                
                if x == "softmax_temp":
                    namex = "softmax_score_temp_"
                else:
                    namex = x + "_score_"
                v1_in = getattr(self,  namex + "in")
                v1_out = getattr(self,  namex + "out")

                if y == "softmax_temp":
                    namey = "softmax_score_temp_"
                else:
                    namey = y + "_score_"
                v2_in = getattr(self,  namey + "in")
                v2_out = getattr(self,  namey + "out")


                score_in = np.empty_like(v1_in)
                for i, _ in enumerate(v1_in):
                    v_sum = v1_in[i] + v2_in[i]
                    if v_sum == 2:
                        score_in[i] = 1
                    elif v_sum == 0:
                        score_in[i] = 0
                    else:
                        v1_score = getattr(score_copy,  namex + "in")
                        v2_score = getattr(score_copy,  namey + "in")

                        t1 = getattr(tresholds, x + "_t")
                        t2 = getattr(tresholds, y + "_t")

                        #normalisieren
                        v1_score = 1/t1 * v1_score[i]
                        v2_score = 1/t2 * v2_score[i]

                        if abs(1-v1_score) >= abs(1-v2_score):
                            score_in[i] = v1_in[i]
                        else:
                            score_in[i] = v2_in[i]

                            



                score_out = np.empty_like(v1_out)
                for i, _ in enumerate(v1_out):
                    v_sum = v1_out[i] + v2_out[i]
                    if v_sum == 2:
                        score_out[i] = 1
                    elif v_sum == 0:
                        score_out[i] = 0
                    else:
                        v1_score = getattr(score_copy,  namex + "out")
                        v2_score = getattr(score_copy,  namey + "out")
                        t1 = getattr(tresholds, x + "_t")
                        t2 = getattr(tresholds, y + "_t")

                        #normalisieren
                        v1_score = 1/t1 * v1_score[i]
                        v2_score = 1/t2 * v2_score[i]

                        if abs(1-v1_score) >= abs(1-v2_score):
                            score_out[i] = v1_out[i]
                        else:
                            score_out[i] = v2_out[i]


                hlist_kombi.append(x + "_" + y)
                hlist_in.append(score_in)
                hlist_out.append(score_out)

                
        randoms = random.sample(range(0, len(hlist_kombi)), 5)
        for r in randoms:
            list_kombi.append(hlist_kombi[r])
            list_in.append(hlist_in[r])
            list_out.append(hlist_out[r])


        #3er Kombi



        methods1 = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha", "doc"]
        methods2 = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha", "doc"]
        methods3 = ["logits", "varianz", "softmax", "softmax_temp", "cosine","energy", "entropy", "gda_maha", "gda_eucl", "maha", "doc"]

        hlist_kombi = []
        hlist_in = []
        hlist_out = []

        for i, x in enumerate(methods1):
    
            methods2.pop(0)
            methods3.pop(0)
            m3Help = copy.deepcopy(methods3)
            for y in methods2:
                m3Help.pop(0)
                for z in m3Help:

                    if x == "softmax_temp":
                        namex = "softmax_score_temp_"
                    else:
                        namex = x + "_score_"
                    v1_in = getattr(self,  namex + "in")
                    v1_out = getattr(self,  namex + "out")

                    if y == "softmax_temp":
                        namey = "softmax_score_temp_"
                    else:
                        namey = y + "_score_"
                    v2_in = getattr(self,  namey + "in")
                    v2_out = getattr(self,  namey + "out")

                    if z == "softmax_temp":
                        namez = "softmax_score_temp_"
                    else:
                        namez = z + "_score_"
                    v3_in = getattr(self,  namez + "in")
                    v3_out = getattr(self,  namez + "out")


                    score_in = np.empty_like(v1_in)
                    for i, _ in enumerate(v1_in):
                        v_sum = v1_in[i] + v2_in[i] + v3_in[i]
                        if v_sum >= 2:
                            score_in[i] = 1
                        else:
                            score_in[i] = 0


                    score_out = np.empty_like(v1_out)
                    for i, _ in enumerate(v1_out):
                        v_sum = v1_out[i] + v2_out[i] + v3_out[i]
                        if v_sum < 2:
                            score_out[i] = 0
                        else:
                            score_out[i] = 1

                    hlist_kombi.append(x + "_" + y + "_" + z)
                    hlist_in.append(score_in)
                    hlist_out.append(score_out)

        randoms = random.sample(range(0, len(hlist_kombi)), 5)
        for r in randoms:
            list_kombi.append(hlist_kombi[r])
            list_in.append(hlist_in[r])
            list_out.append(hlist_out[r])

        return list_kombi, list_in, list_out

          

                


    def apply_ocsvm(self, args, method):

        def get_ocsvm_scores(ocsvm_train, ocsvm_in, ocsvm_out):
            
            if not isinstance(ocsvm_train, np.ndarray):
                ocsvm_train = ocsvm_train.cpu().detach().numpy()
            if not isinstance(ocsvm_in, np.ndarray):
                ocsvm_in = ocsvm_in.cpu().detach().numpy()
            if not isinstance(ocsvm_out, np.ndarray):
                ocsvm_out = ocsvm_out.cpu().detach().numpy()
            #SVM
            steps = np.linspace(0.01, 0.99, 100)


            best_t = 0
            best_acc = 0
            for i in steps:
                #erstmal weglassen, ist ja eigentlich auch nicht richtig
                break
                nuu = i
                c_lr = None
                c_lr = svm.OneClassSVM(nu=nuu, kernel='linear', degree=2)
                #c_lr = svm.OneClassSVM(gamma='scale', nu=0.01)
                c_lr.fit(ocsvm_train)

                pred_in = c_lr.predict(ocsvm_in)
                pred_in = np.where(pred_in == -1, 0, 1)

                pred_out = c_lr.predict(ocsvm_out)
                pred_out = np.where(pred_out == -1, 0, 1)

                labels_in = np.ones_like(pred_in).astype(np.int64)
                labels_out = np.zeros_like(pred_out).astype(np.int64)
                labels_gesamt = np.concatenate((labels_in, labels_out), axis=-1)

                t_pred = np.concatenate((pred_in, pred_out), axis=-1)

                acc = accuracy_score(labels_gesamt, t_pred)
                if acc > best_acc:
                    best_acc = acc
                    best_t = i

            nuu = 0.8
            c_lr = None
            c_lr = svm.OneClassSVM(nu=nuu, kernel='linear', degree=2)
            #c_lr = svm.OneClassSVM(gamma='scale', nu=0.01)
            c_lr.fit(ocsvm_train)

            if method == "predict":
                #return 0|1
                pred_in = c_lr.predict(ocsvm_in)
                pred_in = np.where(pred_in == -1, 0, 1)
                pred_out = c_lr.predict(ocsvm_out)
                pred_out = np.where(pred_out == -1, 0, 1)
            else:
                #return scores optimiert von ocsvm
                pred_in = c_lr.score_samples(ocsvm_in)
                pred_out = c_lr.score_samples(ocsvm_out)

            return pred_in, pred_out


        ################## LOGITS ########################
        self.logits_score_in_ocsvm, self.logits_score_out_ocsvm = get_ocsvm_scores(self.logits_score_train_ocsvm, self.logits_score_in_ocsvm, self.logits_score_out_ocsvm)
        ################## SOFTMAX ########################
        self.softmax_score_in_ocsvm, self.softmax_score_out_ocsvm = get_ocsvm_scores(self.softmax_score_train_ocsvm, self.softmax_score_in_ocsvm, self.softmax_score_out_ocsvm)
        ################## SOFTMAX TEMP ###########################
        self.softmax_score_temp_in_ocsvm, self.softmax_score_temp_out_ocsvm = get_ocsvm_scores(self.softmax_score_temp_train_ocsvm, self.softmax_score_temp_in_ocsvm, self.softmax_score_temp_out_ocsvm)
        ################### MAHA ############################
        self.maha_score_in_ocsvm, self.maha_score_out_ocsvm = get_ocsvm_scores(self.maha_score_dev_ocsvm, self.maha_score_in_ocsvm, self.maha_score_out_ocsvm)
        ################## Cosine ###############
        self.cosine_score_in_ocsvm, self.cosine_score_out_ocsvm = get_ocsvm_scores(self.cosine_score_dev_ocsvm, self.cosine_score_in_ocsvm, self.cosine_score_out_ocsvm)
        ################### GDA ###############
        if args.model_ID != 14:
            self.gda_maha_score_in_ocsvm, self.gda_maha_score_out_ocsvm = get_ocsvm_scores(self.gda_maha_score_train_ocsvm, self.gda_maha_score_in_ocsvm, self.gda_maha_score_out_ocsvm)
            self.gda_eucl_score_in_ocsvm, self.gda_eucl_score_out_ocsvm = get_ocsvm_scores(self.gda_eucl_score_dev_ocsvm, self.gda_eucl_score_in_ocsvm, self.gda_eucl_score_out_ocsvm)

    def apply_tresholds(self, args, tresholds, pooled_in = None, pooled_out = None, centroids = None, delta = None):


        ################## LOGITS ########################
        self.logits_score_in = np.where(self.logits_score_in >= tresholds.logits_t, 1, 0)
        self.logits_score_out = np.where(self.logits_score_out >= tresholds.logits_t, 1, 0)

        ################## VARIANZ ########################
        self.varianz_score_in = np.where(self.varianz_score_in >= tresholds.varianz_t, 1, 0)
        self.varianz_score_out = np.where(self.varianz_score_out >= tresholds.varianz_t, 1, 0)

        ################## SOFTMAX ########################
        self.softmax_score_in = np.where(self.softmax_score_in >= tresholds.softmax_t, 1, 0)
        self.softmax_score_out = np.where(self.softmax_score_out >= tresholds.softmax_t, 1, 0)
    
        ################## SOFTMAX TEMP ###########################
        self.softmax_score_temp_in = np.where(self.softmax_score_temp_in >= tresholds.softmax_temp_t, 1, 0)
        self.softmax_score_temp_out = np.where(self.softmax_score_temp_out >= tresholds.softmax_temp_t, 1, 0)

        ################### MAHA ############################
        self.maha_score_in = np.where(self.maha_score_in <= tresholds.maha_t, 1, 0)
        self.maha_score_out = np.where(self.maha_score_out <= tresholds.maha_t, 1, 0)

        ################## Entropy ###############
        self.entropy_score_in = np.where(self.entropy_score_in <= tresholds.entropy_t, 1, 0)
        self.entropy_score_out = np.where(self.entropy_score_out <= tresholds.entropy_t, 1, 0)

        ################## Cosine ###############
        self.cosine_score_in = np.where(self.cosine_score_in >= tresholds.cosine_t, 1, 0)
        self.cosine_score_out = np.where(self.cosine_score_out >= tresholds.cosine_t, 1, 0)

        ################## Energy ###############
        self.energy_score_in = np.where(self.energy_score_in >= tresholds.energy_t, 1, 0)
        self.energy_score_out = np.where(self.energy_score_out >= tresholds.energy_t, 1, 0)

        if args.model_ID != 14:
            ################### GDA ###############
            self.gda_maha_score_in = np.where(self.gda_maha_score_in <= tresholds.gda_maha_t, 1, 0)
            self.gda_maha_score_out = np.where(self.gda_maha_score_out <= tresholds.gda_maha_t, 1, 0)
            self.gda_eucl_score_in = np.where(self.gda_eucl_score_in <= tresholds.gda_eucl_t, 1, 0)
            self.gda_eucl_score_out = np.where(self.gda_eucl_score_out <= tresholds.gda_eucl_t, 1, 0)

    def ood_classification(self):

        self.logits_idx_in = np.where(self.logits_idx_in > 0, 1, 0)
        self.logits_idx_out = np.where(self.logits_idx_out > 0, 1, 0)
        self.softmax_temp_idx_in = np.where(self.softmax_temp_idx_in > 0, 1, 0)
        self.softmax_temp_idx_out = np.where(self.softmax_temp_idx_out > 0, 1, 0)
        self.cosine_idx_in = np.where(self.cosine_idx_in > 0, 1, 0)
        self.cosine_idx_out = np.where(self.cosine_idx_out > 0, 1, 0)
        self.maha_idx_in = np.where(self.maha_idx_in > 0, 1, 0)
        self.maha_idx_out = np.where(self.maha_idx_out > 0, 1, 0)



################################################ Tresholds ##################################################################
class Tresholds():

    def __init__(self):
        
        self.logits_t = 0
        self.softmax_t = 0
        self.softmax_temp_t = 0
        self.maha_t = 0
        self.entropy_t = 0
        self.cosine_t = 0
        self.energy_t = 0
        self.gda_maha_t = 0
        self.gda_eucl_t = 0


    def treshold_picker(self, args, method, pred_dev, pred_in, pred_out, f, t, step, min):

        if method == "avg":
            if min is False:
                
                # n-Niedrgiste Wert -> n z.b. so wählen, dass 90 % drin sind
                n = int(pred_dev.size/10)
                t =  np.partition(pred_dev, n-1)[n-1]

                #niedrigster Wert
                #t =  np.amin(pred_dev)

                #Mean - Varianz
                #t = np.mean(pred_dev) - np.var(pred_dev)

                #max - 1/3 der Distanz zwischen max und mean
                #t = np.max(pred_dev) - (np.max(pred_dev) - np.mean(pred_dev)) * 1/3
                return t

            else:

                # n-Hächste Wert -> n z.b. so wählen, dass 90 % drin sind
                n = int(int(pred_dev.size) - int(pred_dev.size/10))
                t = np.partition(pred_dev, n-1)[n-1]

                #Max Wert
                #t = np.max(pred_dev)

                #Mean - Varianz
                #t = np.mean(pred_dev) - np.var(pred_dev)

                #max - 1/x der Distanz zwischen max und mean
                #t = np.min(pred_dev) + (np.mean(pred_dev) - np.min(pred_dev)) * 1/3

                return t

        elif 'best' in method:

            if not isinstance(pred_in, np.ndarray):
                pred_in = pred_in.cpu().detach().numpy()
            if not isinstance(pred_out, np.ndarray):
                pred_out = pred_out.cpu().detach().numpy()

            if method == "best_dev":
                pred_in = pred_dev
                pred_out = pred_dev

            #WENN NACH F1 optimiert werden soll:
            #Treshold berechnen von ID3
            #https://github.com/parZival27/supervised-contrastive-learning-for-out-of-domain-detection/blob/358c6069712a1966a65fb06c3ba43cf8f8239dca/utils.py#L223
            #nehmen fp, tp und fn = 0

                if min is False: 
                    seen_m_dist = pred_out
                    unseen_m_dist = pred_in
                else:
                    seen_m_dist = pred_in
                    unseen_m_dist = pred_out
                    
                lst = []
                for item in seen_m_dist:
                    lst.append((item, "seen"))
                for item in unseen_m_dist:
                    lst.append((item, "unseen"))
                # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
                lst = sorted(lst, key=lambda item: item[0])

                threshold = 0.
                tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0

                def compute_f1(tp, fp, fn):
                    p = tp / (tp + fp + 1e-10)
                    r = tp / (tp + fn + 1e-10)
                    return (2 * p * r) / (p + r + 1e-10)

                f1 = compute_f1(tp, fp, fn)

                for m_dist, label in lst:
                    if label == "seen":  # fp -> tn
                        fp -= 1
                    else:  # tp -> fn
                        tp -= 1
                        fn += 1
                    if compute_f1(tp, fp, fn) > f1:
                        f1 = compute_f1(tp, fp, fn)
                        threshold = m_dist + 1e-10

                print("estimated threshold:", threshold)

            else:
                

                labels_in = np.ones_like(pred_in).astype(np.int64)
                labels_out = np.zeros_like(pred_out).astype(np.int64)
                
                #WENN NACH ACC+REC OPTIMIERT WERDEN SOLL
                threshold = 0
                best_acc = 0
                

                steps = np.linspace(f,t,step)
                for i in steps:

                    t_pred = np.concatenate((pred_in, pred_out), axis=-1)
                    if min is True:
                        t_pred = np.where(t_pred <= i, 1, 0)
                        t_pred_in =  np.where(pred_in <= i, 1, 0)
                        t_pred_out = np.where(pred_out <= i, 1, 0) 
                    else:
                        t_pred_in =  np.where(pred_in > i, 1, 0)
                        t_pred_out = np.where(pred_out > i, 1, 0)
                    acc = accuracy_score(labels_in, t_pred_in)
                    rec = recall_score(labels_out, t_pred_out, pos_label=0)

                    if acc+rec > best_acc:
                        best_acc = acc+rec
                        threshold = i
                

            return threshold


    def calculate_tresholds(self, args, scores, method):

        #Todo: hier anhand von dev oder Train???

        ################## LOGITS ########################
        self.logits_t = self.treshold_picker(args, method, scores.logits_score_dev, scores.logits_score_in, scores.logits_score_out, np.min(scores.logits_score_out), max(scores.logits_score_out), 500, min=False)
        ################## VARIANZ ########################
        self.varianz_t = self.treshold_picker(args, method, scores.varianz_score_dev, scores.varianz_score_in, scores.varianz_score_out, np.min(scores.varianz_score_out), max(scores.varianz_score_in), 500, min=False)

        ################## SOFTMAX ########################
        self.softmax_t = self.treshold_picker(args, method, scores.softmax_score_dev, scores.softmax_score_in, scores.softmax_score_out, np.min(scores.softmax_score_out), max(scores.softmax_score_in), 500, min=False)
    
        ################## SOFTMAX TEMP ###########################
        self.softmax_temp_t = self.treshold_picker(args, method, scores.softmax_score_temp_dev, scores.softmax_score_temp_in ,scores.softmax_score_temp_out ,np.min(scores.softmax_score_temp_out), max(scores.softmax_score_temp_in), 500, min=False)

        ################### MAHA ############################
        self.maha_t = self.treshold_picker(args, method, scores.maha_score_dev, scores.maha_score_in, scores.maha_score_out, np.min(scores.maha_score_in), max(scores.maha_score_out), 500, min=True)

        ################## Entropy ###############
        self.entropy_t = self.treshold_picker(args, method, scores.entropy_score_dev, scores.entropy_score_in, scores.entropy_score_out, np.min(scores.entropy_score_in), max(scores.entropy_score_out), 500, min=True)

        ################## Cosine ###############
        self.cosine_t = self.treshold_picker(args, method, scores.cosine_score_dev, scores.cosine_score_in, scores.cosine_score_out, np.min(scores.cosine_score_out), max(scores.cosine_score_in), 500, min=False)

        ################## Energy ###############
        self.energy_t = self.treshold_picker(args, method, scores.energy_score_dev, scores.energy_score_in, scores.energy_score_out, np.min(scores.energy_score_out), max(scores.energy_score_in), 500, min=False)
        
        if args.model_ID != 14:
            ################### GDA ###############
            self.gda_eucl_t = self.treshold_picker(args, method, scores.gda_eucl_score_dev, scores.gda_eucl_score_in, scores.gda_eucl_score_out, np.min(scores.gda_eucl_score_in), max(scores.gda_eucl_score_out), 500, min=True)
            self.gda_maha_t = self.treshold_picker(args, method, scores.gda_maha_score_dev, scores.gda_maha_score_in, scores.gda_maha_score_out, np.min(scores.gda_maha_score_in), max(scores.gda_maha_score_out), 500, min=True)

    # def adb_treshold(self, preds, pooled, centroids, delta):
    #     preds = torch.tensor(preds, dtype=torch.long).to('cuda:0')
    #     preds_ones = torch.ones_like(preds)
    #     euc_dis = torch.norm(pooled - centroids[preds], 2, 1).view(-1)           
    #     preds_ones[euc_dis >= delta[preds]] = 0
    #     return preds_ones.detach().cpu().numpy()

#- Evaluate




###########################
#FÜR ID 0 mit BERT
def detect_ood_bert(args, model, test_id_dataset, test_ood_dataset, best_temp=None):


    def compute_ood_outputs(model, input_ids=None, attention_mask=None):

        outputs = model.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits, pooled = model.classifier(sequence_output)

        #outputs = model.bert(input_ids, attention_mask=attention_mask)
        #pooled_output = pooled = outputs[1]
        #pooled_output = model.dropout(pooled_output)
        #logits = model.classifier(pooled_output)
        

        return logits

    all_logits_in = []
    all_logits_out = []
    #Test-ID:
    for batch in tqdm(test_id_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            all_logits_in.append(compute_ood_outputs(model, input_ids, attention_mask))
        
    all_logits_in = reduce(lambda x,y: torch.cat((x,y)), all_logits_in[::])

    
    #Test-OOD:
    for batch in tqdm(test_ood_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            all_logits_out.append(compute_ood_outputs(model, input_ids, attention_mask))
           
    all_logits_out = reduce(lambda x,y: torch.cat((x,y)), all_logits_out[::])


    logits_in = get_model_prediction(all_logits_in)
    logits_out = get_model_prediction(all_logits_out)
    softmax_in = get_softmax_score(all_logits_in, False)
    softmax_out = get_softmax_score(all_logits_out, False)


    def bert_treshold(pred_in, pred_out, f,t,step, min=False):
    
        #WENN NACH ACC+REC OPTIMIERT WERDEN SOLL
        threshold = 0
        best_acc = 0
        labels_in = np.ones_like(pred_in).astype(np.int64)
        labels_out = np.zeros_like(pred_out).astype(np.int64)
        labels_gesamt = np.concatenate((labels_in, labels_out), axis=-1)

        steps = np.linspace(f,t,step)
        for i in steps:

            t_pred = np.concatenate((pred_in, pred_out), axis=-1)
            if min is True:
                t_pred = np.where(t_pred <= i, 1, 0)
                t_pred_in =  np.where(pred_in <= i, 1, 0)
                t_pred_out = np.where(pred_out <= i, 1, 0) 
            else:
                t_pred_in =  np.where(pred_in > i, 1, 0)
                t_pred_out = np.where(pred_out > i, 1, 0)
            acc = accuracy_score(labels_in, t_pred_in)
            rec = recall_score(labels_out, t_pred_out, pos_label=0)

            if acc+rec > best_acc:
                best_acc = acc+rec
                threshold = i

        return best_acc

    logits_acc = bert_treshold(logits_in, logits_out, 0, 10, 1000)
    print(logits_acc)
    softmax_acc = bert_treshold(softmax_in, softmax_out, 0, 10, 1000)
    print(softmax_acc)