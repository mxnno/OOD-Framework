import torch
import numpy as np
import wandb
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
from evaluation import evaluate_metriken_ohne_Treshold, evaluate_mit_Treshold, evaluate_scores_ohne_Treshold, evaluate_NLI, evaluate_ADB
from scipy.stats import chi2
from model import  set_model


#Paper -> soll mit in die Arbeit


def detect_ood(args, model, train_dataset, dev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None, best_temp=None):
    

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
    for batch in tqdm(dev_id_dataset):
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
        #detect_ood_adb(args, centroids, delta, all_pool_in, all_pool_out)
        #return
        all_logits_train, all_logits_dev, all_logits_in , all_logits_out = detect_ood_adb2(centroids, delta, all_pool_in, all_pool_out, all_pool_dev, all_pool_train)


#2. Treshold + Scores
    
    #scores_dev = Scores(thresholds, all_logits_dev, all_logits_out, all_pool_dev, all_pool_out, all_logits_train, all_pool_train, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    #scores_dev.calculate_scores(best_temp)
    #

    scores = Scores(all_logits_in, all_logits_out, all_pool_in, all_pool_out, all_logits_train, all_pool_train, all_logits_dev, all_pool_dev, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    print("Calculate all scores...")
    scores.calculate_scores(best_temp, args)
    
    
#     # (muss als erstes)
# # 2.1 Metriken ohne Treshold -> man braucht die Scores, nicht 0,1 ...
#     # -> nicht für LOF und DOC möglich
#     # -> Maha nur in Kombi mit OC-SVM-Scores
#     # --> bei softmax wesentlich schlechtere Erg mit OC-SVM -> softmax mit und ohne OC-SVM Scores in einer Tabelle
#     print("Metriken ohne Treshold...")
#     scores_ocsvm = deepcopy(scores)
#     scores_ocsvm.apply_ocsvm(args, "scores")
#     #-> scores_in bzw. scors_out UND scores_in_ocsvm bzw. scores_out_ocsvm in eval abfragen!
#     evaluate_metriken_ohne_Treshold(args, scores_ocsvm)



# 2.2 mit Treshold (best. avg)

    print("Mit Treshold...")
    print("...best...")
    scores_best = deepcopy(scores)
    #thresholds.calculate_tresholds(args, scores_best, 'best')
    scores_best.apply_tresholds(all_pool_in, all_pool_out, centroids, delta)
    evaluate_mit_Treshold(args, scores_best, 'best')
    print("...best_dev...")
    scores_best_dev = deepcopy(scores)
    #thresholds.calculate_tresholds(args, scores_best_dev, 'best_dev')
    scores_best_dev.apply_tresholds(all_pool_in, all_pool_out, centroids, delta)
    evaluate_mit_Treshold(args, scores_best_dev, 'best_dev')
    print("...avg...")
    scores_avg = deepcopy(scores)
    #thresholds.calculate_tresholds(args, scores_avg, 'avg')
    scores_avg.apply_tresholds(all_pool_in, all_pool_out, centroids, delta)
    evaluate_mit_Treshold(args, scores_avg, 'avg')
    
# # 2.2 ohne Treshold zu 0/1
#     # - OCSVM Predict (logits, softmax ...)
#     # - DOC (logits) -> geht nur für alleine
#     # - LOF (logits+pool) -> geht nur für alleine
#     print("Scores ohne Treshold...")
#     #für ADB 
    
#     scores_ohne_Treshold = deepcopy(scores)
#     scores_ohne_Treshold.apply_ocsvm(args, 'predict')
#     evaluate_scores_ohne_Treshold(args, scores_ohne_Treshold)


   
def detect_ood_adb(args, centroids, delta, pool_in, pool_out):

    
    def get_adb_score(pooled, centroids, delta):

        logits_adb = euclidean_metric(pooled, centroids)
        #qwert
        #Kann ich die logits rausnehmen für anderen Methoden???
        # -> heir ja nur Softmax
        probs, preds = F.softmax(logits_adb.detach(), dim = 1).max(dim = 1)
        preds_ones = torch.ones_like(preds)
        preds.to('cuda:0')
        euc_dis = torch.norm(pooled - centroids[preds], 2, 1).view(-1)           
        preds_ones[euc_dis >= delta[preds]] = 0
        return preds_ones.cpu().detach().numpy()

    adb_pred_in = get_adb_score(pool_in, centroids, delta)
    adb_pred_out = get_adb_score(pool_out, centroids, delta)

    evaluate_ADB(args, adb_pred_in, adb_pred_out)

def detect_ood_adb2(centroids, delta, pool_in, pool_out, pool_dev, pool_train):

    logits_train = euclidean_metric(pool_train, centroids)
    logits_dev = euclidean_metric(pool_dev, centroids)
    logits_in = euclidean_metric(pool_in, centroids)
    logits_out = euclidean_metric(pool_out, centroids)

    return logits_train, logits_dev, logits_in, logits_out







#################################################### METHODEN ####################################################



def get_model_prediction(logits, full_score=False):
    #return label_pred des Models
    if full_score:
        return logits

    pred, label_pred = logits.max(dim = 1)
    return label_pred


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
    return idx


def get_softmax_score_with_temp(logits, temp, full_score=False):
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


    softmax, idx = F.softmax((logits / temp), dim=-1).max(-1)
    return idx



def get_maha_score(pooled, all_classes, class_mean, class_var, full_scores=False):

    #ID 2:
    maha_score = []
    for c in all_classes:
        centered_pooled = pooled - class_mean[c].unsqueeze(0)
        ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
        maha_score.append(ms)
    maha_score_full = torch.stack(maha_score, dim=-1)
    if full_scores is True:
        return maha_score_full
    maha_score = maha_score_full.min(-1)[1]
    #maha_score = -maha_score
    maha_score = maha_score
    return maha_score
    #https://www.statology.org/mahalanobis-distance-python/
    # aus distanz ein p-Value berechnen -> outlier



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

        
        self.softmax_score_in_ocsvm = 0
        self.softmax_score_out_ocsvm = 0
        self.softmax_score_train_ocsvm = 0
        self.softmax_score_train = 0
        self.softmax_score_dev = 0
        self.softmax_score_in = 0
        self.softmax_score_out = 0

        self.softmax_score_temp_in_ocsvm = 0
        self.softmax_score_temp_out_ocsvm = 0
        self.softmax_score_temp_train_ocsvm = 0
        self.softmax_score_temp_train = 0
        self.softmax_score_temp_dev = 0
        self.softmax_score_temp_in = 0
        self.softmax_score_temp_out = 0

        self.cosine_score_in_ocsvm = 0
        self.cosine_score_out_ocsvm = 0
        self.cosine_score_train_ocsvm = 0
        self.cosine_score_train = 0
        self.cosine_score_dev = 0
        self.cosine_score_in = 0
        self.cosine_score_out = 0


        self.maha_score_in_ocsvm = 0
        self.maha_score_out_ocsvm = 0
        self.maha_score_dev_ocsvm = 0
        self.maha_score_train = 0
        self.maha_score_dev = 0
        self.maha_score_in = 0
        self.maha_score_out = 0


    def calculate_scores(self, best_temp, args=None):


        ################## Nur Logits #####################
        self.logits_score_train = get_model_prediction(self.logits_train, False)
        self.logits_score_dev = get_model_prediction(self.logits_dev, False)
        self.logits_score_in = get_model_prediction(self.logits_in, False)
        self.logits_score_out = get_model_prediction(self.logits_out, False)


        ################## SOFTMAX ########################
        self.softmax_score_train = get_softmax_score(self.logits_train, False)
        self.softmax_score_dev = get_softmax_score(self.logits_dev, False)
        self.softmax_score_in = get_softmax_score(self.logits_in, False)
        self.softmax_score_out= get_softmax_score(self.logits_out, False)

        ################# SOFTMAX TMP #####################
        self.softmax_score_temp_train = get_softmax_score_with_temp(self.logits_train, best_temp, False)
        self.softmax_score_temp_dev = get_softmax_score_with_temp(self.logits_dev, best_temp, False)
        self.softmax_score_temp_in = get_softmax_score_with_temp(self.logits_in, best_temp, False)
        self.softmax_score_temp_out = get_softmax_score_with_temp(self.logits_out, best_temp, False)



        ################### MAHA ############################
        self.maha_score_train = get_maha_score(self.pooled_train, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_dev = get_maha_score(self.pooled_dev, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_in = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_out = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, False)


    
    def apply_tresholds(self, pooled_in = None, pooled_out = None, centroids = None, delta = None):

        self.logits_score_in = adb_treshold(self.logits_score_in, pooled_in, centroids, delta)
        self.logits_score_out = adb_treshold(self.logits_score_out, pooled_out, centroids, delta)
        ################## SOFTMAX ########################
        self.softmax_score_in = adb_treshold(self.softmax_score_in, pooled_in, centroids, delta)
        self.softmax_score_out = adb_treshold(self.softmax_score_out, pooled_out, centroids, delta)
        ################## SOFTMAX TEMP ###########################
        self.softmax_score_temp_in = adb_treshold(self.softmax_score_temp_in, pooled_in, centroids, delta)
        self.softmax_score_temp_out = adb_treshold(self.softmax_score_temp_out, pooled_out, centroids, delta)
        ################### MAHA ############################
        self.maha_score_in = adb_treshold(self.maha_score_in, pooled_in, centroids, delta)
        self.maha_score_out = adb_treshold(self.maha_score_out, pooled_out, centroids, delta)
        ################## Cosine ###############
        #self.cosine_score_in = adb_treshold(self.cosine_score_in, pooled_in, centroids, delta)
        #self.cosine_score_out = adb_treshold(self.cosine_score_out, pooled_out, centroids, delta)
    
def adb_treshold(preds, pooled, centroids, delta):
    a = torch.device('cuda:0')
    new_preds = preds.to(a, dtype=torch.long)
    preds_ones = torch.ones_like(new_preds)
    euc_dis = torch.norm(pooled - centroids[new_preds], 2, 1).view(-1)           
    preds_ones[euc_dis >= delta[new_preds]] = 0
    return preds_ones.cpu().detach().numpy()

#- Evaluate



