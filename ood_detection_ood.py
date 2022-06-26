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


    softmax_in = get_softmax_score(all_logits_in, full_score=True)
    softmax_out = get_softmax_score(all_logits_out, full_score=True)
    softmax_in = softmax_in.cpu().detach().numpy()
    softmax_out = softmax_out.cpu().detach().numpy()

    all_logits_in = all_logits_in.cpu().detach().numpy()
    all_logits_out = all_logits_out.cpu().detach().numpy()


    print("###### LOGITS ######")
    v1(all_logits_in, all_logits_out)
    v2(all_logits_in, all_logits_out)
    v3(all_logits_in, all_logits_out)

    print("###### SOFTMAX ######")
    v1(softmax_in, softmax_out)
    v2(softmax_in, softmax_out)
    v3(softmax_in, softmax_out)



def v1(logits_score_in, logits_score_out):
    #Variante 1: 0 > 1
    idx_in = logits_score_in.argmax(axis = -1)
    idx_out = logits_score_out.argmax(axis = -1)

    pred_in = np.where(idx_in > 0, 1, 0)
    pred_out = np.where(idx_out > 0, 1, 0)

    labels_in = np.ones_like(pred_in).astype(np.int64)
    labels_out = np.zeros_like(pred_out).astype(np.int64)

    in_acc = accuracy_score(labels_in, pred_in)
    out_acc = accuracy_score(labels_out, pred_out)

    v1 = in_acc + out_acc
    print("V_org: " + str(v1))


def v2(logits_score_in, logits_score_out):

    #Variante 2: treshold ID

    t_2 = get_trehsold_with_ood_id(logits_score_in, logits_score_out, 500, False)

    #labeled:
    pred_in =  np.where(logits_score_in[:,1:].max(axis = 1) > t_2, 1, 0)
    pred_out =  np.where(logits_score_out[:,1:].max(axis = 1) > t_2, 1, 0)

    labels_in = np.ones_like(pred_in).astype(np.int64)
    labels_out = np.zeros_like(pred_out).astype(np.int64)

    in_acc = accuracy_score(labels_in, pred_in)
    out_acc = accuracy_score(labels_out, pred_out)

    v2 = in_acc + out_acc
    print("V_T_ID: " + str(v2))

def v3(logits_score_in, logits_score_out):
    #Variante 3: treshold OOD

    t_3 = get_trehsold_with_ood_ood(logits_score_in, logits_score_out, 500, False)

    #labeled:
    pred_in =  np.where(logits_score_in[:,0] > t_3, 0, 1)
    pred_out =  np.where(logits_score_out[:,0] > t_3, 0, 1)

    #Validation
    labels_in = np.ones_like(pred_in).astype(np.int64)
    labels_out = np.zeros_like(pred_out).astype(np.int64)

    #labeled:
    in_acc = accuracy_score(labels_in, pred_in)
    out_acc = accuracy_score(labels_out, pred_out)

    v3 = in_acc + out_acc
    print("V_T_OOD: " + str(v3))


#################################################### METHODEN ####################################################



def get_model_prediction(logits, full_score=False):
    #return label_pred des Models
    if full_score:
        return logits

    pred, label_pred = logits.max(dim = 1)
    return label_pred.cpu().detach().numpy()

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
    return idx.cpu().detach().numpy()


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
    return idx.cpu().detach().numpy()

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

def get_cosine_score(pooled, norm_bank, full_scores=False):
    #von ID 2
    #hier fehlt noch norm_bank
    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = norm_pooled @ norm_bank.t()
    if full_scores is True:
        return cosine_score
    cosine_score = cosine_score.max(-1)[1]
    return cosine_score.cpu().detach().numpy()


def get_energy_score(logits):
    #-> hier kein Full_score möglich -> nicht für OCSVM geeignet...

    #von ID 2
    return torch.logsumexp(logits, dim=-1).cpu().detach().numpy()


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
    maha_score = maha_score.cpu().detach().numpy()
    return maha_score
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
    lof = LocalOutlierFactor(n_neighbors=args.few_shot, contamination = 0.5, n_jobs=-1)
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



def get_trehsold_with_ood_ood(logits_in, logits_out, step, min):
    
    t = logits_out[:,0].max()
    f = logits_in[:,0].min()
    threshold = 0
    best_acc = 0
    labels_in = np.ones_like(logits_in.max(axis = 1)).astype(np.int64)
    labels_out = np.zeros_like(logits_out.max(axis = 1)).astype(np.int64)

    steps = np.linspace(f,t,step)
    for i in steps:

        if min is True:
            t_pred_in =  np.where(logits_in[:,0] <= i, 0, 1)
            t_pred_out = np.where(logits_out[:,0] <= i, 0, 1) 
        else:
            t_pred_in =  np.where(logits_in[:,0] > i, 0, 1)
            t_pred_out = np.where(logits_out[:,0] > i, 0, 1)

        acc = accuracy_score(labels_in, t_pred_in)
        rec = recall_score(labels_out, t_pred_out, pos_label=0)

        if acc+rec > best_acc:
            best_acc = acc+rec
            threshold = i
            

    return threshold

def get_trehsold_with_ood_id(logits_in, logits_out, step, min):
    
    t = logits_in[:,1:].max()
    f = logits_out[:,1:].min()
    threshold = 0
    best_acc = 0
    labels_in = np.ones_like(logits_in.max(axis = 1)).astype(np.int64)
    labels_out = np.zeros_like(logits_out.max(axis = 1)).astype(np.int64)

    steps = np.linspace(f,t,step)
    for i in steps:

        if min is True:
            t_pred_in =  np.where(logits_in[:,1:].max(axis = 1) <= i, 1, 0)
            t_pred_out = np.where(logits_out[:,1:].max(axis = 1) <= i, 1, 0)
        else:
            t_pred_in =  np.where(logits_in[:,1:].max(axis = 1) > i,1, 0)
            t_pred_out = np.where(logits_out[:,1:].max(axis = 1) > i, 1, 0)

        acc = accuracy_score(labels_in, t_pred_in)
        rec = recall_score(labels_out, t_pred_out, pos_label=0)


        if acc+rec > best_acc:
            best_acc = acc+rec
            threshold = i
            

    return threshold