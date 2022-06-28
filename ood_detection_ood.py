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
from evaluation import evaluate_metriken_ohne_Treshold, evaluate_mit_Treshold, evaluate_scores_ohne_Treshold, evaluate_NLI, evaluate_ADB, evaluate_mit_OOD
from scipy.stats import chi2
from model import  set_model


#Paper -> soll mit in die Arbeit


def detect_ood(args, model, train_dataset, dev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None, best_temp=None):
    

    #DETECT OOD wenn es OOD Trainingsdaten gibt

    #1. konservativer Ansatz -> normale Klassifizierung, man nimmt die Klasse mit höchster Prediction

    #2. mit Treshold für ID Daten
    #   - anhand der ID Scores wird ein Treshold gebildet
    #   - Treshold wird auch nur auf ID angewender (np.where(x[:, 1:] > t, 1, 0))
    #   - Idee: falls Modell OOD predicted, dann sind die ID Werte unter dem Treshold und es wird 0 gewählt


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


#2. Treshold + Scores
    
    #scores_dev = Scores(thresholds, all_logits_dev, all_logits_out, all_pool_dev, all_pool_out, all_logits_train, all_pool_train, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    #scores_dev.calculate_scores(best_temp)
    #

    thresholds = Tresholds()
    scores = Scores(all_logits_in, all_logits_out, all_pool_in, all_pool_out, all_logits_train, all_pool_train, all_logits_dev, all_pool_dev, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    print("Calculate all scores...")
    scores.calculate_scores(best_temp, args)


    #Einfach Klassifizierung falls OOD Daten vorhanden -> höchste Klasse gewinnt
    # nur für logits, softmax und maha, da es eh nicht so gut ist -> weniger Aufwand
    scores_ood = deepcopy(scores)
    scores_ood.ood_classification()
    evaluate_mit_OOD(args, scores_ood)
    
    
    # (muss als erstes)
# 2.1 Metriken ohne Treshold -> man braucht die Scores, nicht 0,1 ...
    # -> nicht für LOF und DOC möglich
    # -> Maha nur in Kombi mit OC-SVM-Scores
    # --> bei softmax wesentlich schlechtere Erg mit OC-SVM -> softmax mit und ohne OC-SVM Scores in einer Tabelle
    print("Metriken ohne Treshold...")
    scores_ocsvm = deepcopy(scores)
    scores_ocsvm.apply_ocsvm(args, "scores")
    #-> scores_in bzw. scors_out UND scores_in_ocsvm bzw. scores_out_ocsvm in eval abfragen!
    evaluate_metriken_ohne_Treshold(args, scores_ocsvm)



# 2.2 mit Treshold (best. avg)

    print("Mit Treshold...")
    print("...best...")
    scores_best = deepcopy(scores)
    thresholds.calculate_tresholds(args, scores_best, 'best')
    scores_best.apply_tresholds(args, thresholds, all_pool_in, all_pool_out, centroids, delta)
    evaluate_mit_Treshold(args, scores_best, 'best')
    print("...avg...")
    scores_avg = deepcopy(scores)
    thresholds.calculate_tresholds(args, scores_avg, 'avg')
    scores_avg.apply_tresholds(args, thresholds, all_pool_in, all_pool_out, centroids, delta)
    evaluate_mit_Treshold(args, scores_avg, 'avg')
    
# 2.2 ohne Treshold zu 0/1
    # - OCSVM Predict (logits, softmax ...)
    # - DOC (logits) -> geht nur für alleine
    # - LOF (logits+pool) -> geht nur für alleine
    print("Scores ohne Treshold...")
    #für ADB 
    
    scores_ohne_Treshold = deepcopy(scores)
    scores_ohne_Treshold.apply_ocsvm(args, 'predict')
    evaluate_scores_ohne_Treshold(args, scores_ohne_Treshold)



#################################################### METHODEN ####################################################



def get_model_prediction(logits, full_score=False, idx=False):
    #return label_pred des Models
    if full_score:
        return logits

    pred, label_pred = logits.max(dim = 1)

    if idx:
        pred, label_pred = logits.max(dim = 1)
        return label_pred.cpu().detach().numpy()
    else:
        logits = logits[:,1:]
        pred, label_pred = logits.max(dim = 1)
        return pred.cpu().detach().numpy()


def get_varianz_score(logits):

    var_score = logits.cpu().detach().numpy()
    var_score = var_score[:,1:]
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

    softmax = F.softmax(logits, dim=-1)
    softmax = softmax[:,1:].max(-1)[0]
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

    if idx:
        idx_score = F.softmax((logits / temp), dim=-1).max(-1)[1]
        return idx_score.cpu().detach().numpy()
    else:
        softmax = F.softmax((logits / temp), dim=-1)
        softmax = softmax[:,1:].max(-1)[0]
        return softmax.cpu().detach().numpy()

def get_entropy_score(logits):

    #-> hier kein Full_score möglich -> nicht für OCSVM geeignet...

    #Nach ID 11 (Gold)
    # Input: 
    # - logits eines batches
    # Return :
    # - entropy score

    softmax = F.softmax(logits, dim=-1)
    softmax = softmax[:,1:]
    expo = torch.exp(softmax)
    expo = expo.cpu()
    entro = entropy(expo, axis=1)
    return entro

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
        cosine_score = cosine_score[:,1:].max(-1)[0]
        return cosine_score.cpu().detach().numpy()


def get_energy_score(logits):
    #-> hier kein Full_score möglich -> nicht für OCSVM geeignet...
    energy = torch.logsumexp(logits, dim=-1).cpu().detach().numpy()
    #von ID 2
    #energy = torch.logsumexp(logits[:,1:], dim=-1).cpu().detach().numpy()
    return energy


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
        maha_score = maha_score_full[:,1:].min(-1)[0]
        return maha_score.cpu().detach().numpy()
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

        #falls OOD Daten
        self.logits_idx_in = 0
        self.logits_idx_out = 0
        self.softmax_temp_idx_in = 0
        self.softmax_temp_idx_out = 0
        self.cosine_idx_in = 0
        self.cosine_idx_out = 0
        self.maha_idx_in = 0
        self.maha_idx_out = 0


        
        self.logits_score_in_ocsvm = 0
        self.logits_score_out_ocsvm = 0
        self.logits_score_train_ocsvm = 0
        self.logits_score_train = 0
        self.logits_score_dev = 0
        self.logits_score_in = 0
        self.logits_score_out = 0

        self.varianz_score_train = 0
        self.varianz_score_dev = 0
        self.varianz_score_in = 0
        self.varianz_score_out = 0
        
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

        self.energy_score_train = 0
        self.energy_score_dev = 0
        self.energy_score_in = 0
        self.energy_score_out = 0

        self.entropy_score_train = 0
        self.entropy_score_dev = 0
        self.entropy_score_in = 0
        self.entropy_score_out = 0


        self.maha_score_in_ocsvm = 0
        self.maha_score_out_ocsvm = 0
        self.maha_score_dev_ocsvm = 0
        self.maha_score_train = 0
        self.maha_score_dev = 0
        self.maha_score_in = 0
        self.maha_score_out = 0



    def calculate_scores(self, best_temp, args=None):

        self.logits_idx_in = get_model_prediction(self.logits_in, False, True)
        self.logits_idx_out = get_model_prediction(self.logits_out, False, True)
        self.softmax_temp_idx_in = get_softmax_score_with_temp(self.logits_in, best_temp, False, True)
        self.softmax_temp_idx_out = get_softmax_score_with_temp(self.logits_out, best_temp, False, True)
        self.cosine_idx_in = get_cosine_score(self.pooled_in, self.norm_bank, False, True)
        self.cosine_idx_out = get_cosine_score(self.pooled_out, self.norm_bank, False, True)
        self.maha_idx_in = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, False, True)
        self.maha_idx_out = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, False, True)

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
        self.softmax_score_temp_in_ocsvm = get_softmax_score_with_temp(self.logits_in, best_temp, True)
        self.softmax_score_temp_out_ocsvm = get_softmax_score_with_temp(self.logits_out, best_temp, True)
        self.softmax_score_temp_train_ocsvm = get_softmax_score_with_temp(self.logits_train, best_temp, True)
        self.softmax_score_temp_train = get_softmax_score_with_temp(self.logits_train, best_temp, False)
        self.softmax_score_temp_dev = get_softmax_score_with_temp(self.logits_dev, best_temp, False)
        self.softmax_score_temp_in = get_softmax_score_with_temp(self.logits_in, best_temp, False)
        self.softmax_score_temp_out = get_softmax_score_with_temp(self.logits_out, best_temp, False)

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

       
        ################### MAHA ############################
        self.maha_score_in_ocsvm = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_out_ocsvm = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_dev_ocsvm = get_maha_score(self.pooled_dev, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_train_ocsvm = get_maha_score(self.pooled_train, self.all_classes, self.class_mean, self.class_var, True)
        self.maha_score_train = get_maha_score(self.pooled_train, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_dev = get_maha_score(self.pooled_dev, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_in = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, False)
        self.maha_score_out = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, False)


    def apply_ocsvm(self, args, method):

        def get_ocsvm_scores(ocsvm_train, ocsvm_in, ocsvm_out):
            
            if not isinstance(ocsvm_train, np.ndarray):
                ocsvm_train = ocsvm_train.cpu().detach().numpy()
            if not isinstance(ocsvm_in, np.ndarray):
                ocsvm_in = ocsvm_in.cpu().detach().numpy()
            if not isinstance(ocsvm_out, np.ndarray):
                ocsvm_out = ocsvm_out.cpu().detach().numpy()
            #SVM
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
        self.softmax_score_temp_in = np.where(self.softmax_score_temp_in >= tresholds.sofmtax_temp_t, 1, 0)
        self.softmax_score_temp_out = np.where(self.softmax_score_temp_out >= tresholds.sofmtax_temp_t, 1, 0)

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
        self.sofmtax_temp_t = 0
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
                pass

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
        self.sofmtax_temp_t = self.treshold_picker(args, method, scores.softmax_score_temp_dev, scores.softmax_score_temp_in ,scores.softmax_score_temp_out ,np.min(scores.softmax_score_temp_out), max(scores.softmax_score_temp_in), 500, min=False)

        ################### MAHA ############################
        self.maha_t = self.treshold_picker(args, method, scores.maha_score_dev, scores.maha_score_in, scores.maha_score_out, np.min(scores.maha_score_in), max(scores.maha_score_out), 500, min=True)

        ################## Entropy ###############
        self.entropy_t = self.treshold_picker(args, method, scores.entropy_score_dev, scores.entropy_score_in, scores.entropy_score_out, np.min(scores.entropy_score_in), max(scores.entropy_score_out), 500, min=True)

        ################## Cosine ###############
        self.cosine_t = self.treshold_picker(args, method, scores.cosine_score_dev, scores.cosine_score_in, scores.cosine_score_out, np.min(scores.cosine_score_out), max(scores.cosine_score_in), 500, min=False)

        ################## Energy ###############
        self.energy_t = self.treshold_picker(args, method, scores.energy_score_dev, scores.energy_score_in, scores.energy_score_out, np.min(scores.energy_score_out), max(scores.energy_score_in), 500, min=False)

