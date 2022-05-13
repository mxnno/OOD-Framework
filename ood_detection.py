import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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
from evaluation import evaluate_test
from scipy.stats import chi2
from model import  set_model




def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict



def detect_ood(args, model, train_dataset, dev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None, best_temp=None):
    

    #OOD Detection + Eval in 3 Schritten
    # 1. Model prediction (BATCH SIZE = 1 !!)
    # ToDo: - Varianz + OC-SVM Score (z.B. nach Mahascore zusätzlich) hinzufügen -> Anhand 1-2 Beispielen testen, ob Mehrwert oder nicht
    # 2. Threshold anwenden -> 0 oder 1
    # **************************
    # * - Tresholdunabhängige Metriken 
    # * - best Treshold (über Testdaten) (wie z.B. DNNC oder clinc)
    # * - avg Treshold (über Train/Dev und z.B. über Mean)
    # * - OC-SVM
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

   # für maha etc.
    model.prepare_ood(dev_id_dataset)
    #Wichtig:
    #bei Maha muss prepare und Trehsold predriction auf unterschiedlichen Daten ausgeführt werden, d.h. z.B. prepare mit Training und Treshold mit dev -> sonst kommen immer die gleiche Werte raus (15 oder 90 für alle Maha Predictions)

    #model2 = set_model(args)
    #model.prepare_ood(dev_id_dataset)


    #Train 
    train_labels = []
    for batch in tqdm(train_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        train_label = batch["labels"].cpu().detach()
        with torch.no_grad():
            model.compute_ood_outputs(**batch, centroids=centroids, delta=delta)
        train_labels.append(train_label)
        
    all_logits_train = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_train = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])
    train_labels = reduce(lambda x,y: torch.cat((x,y)), train_labels[::])

    #zum Abspeichern der logits und pools
    #save_logits(all_logits_train, 'all_logits_train.pt')
    #save_logits(all_pool_train, 'all_pool_train.pt')
    model.all_logits = []
    model.all_pool = []

    #Test-ID:
    for batch in tqdm(test_id_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            model.compute_ood_outputs(**batch, centroids=centroids, delta=delta)
        
    all_logits_in = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_in = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])

    #zum Abspeichern der logits und pools
    #save_logits(all_logits_in, '1305_traindev_id_logits.pt')
    print(all_pool_in.size())
    save_logits(all_pool_in, '/content/drive/MyDrive/Masterarbeit/Results/1305_dev_id_pool.pt')
    model.all_logits = []
    model.all_pool = []
    
    #Test-OOD:
    for batch in tqdm(test_ood_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            model.compute_ood_outputs(**batch, centroids=centroids, delta=delta)
           
    all_logits_out = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_out = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])
    #zum Abspeichern der logits und pools
    #save_logits(all_logits_out, '1305_traindev_ood_logits.pt')
    save_logits(all_pool_out, '/content/drive/MyDrive/Masterarbeit/Results/1305_dev_ood_pool.pt')
    print(all_pool_out.size())
    model.all_logits = []
    model.all_pool = []

    #Dev:
    dev_labels = []
    for batch in tqdm(dev_id_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        dev_label = batch["labels"].cpu().detach()
        with torch.no_grad():
            model.compute_ood_outputs(**batch, centroids=centroids, delta=delta)
        dev_labels.append(dev_label)
    
    all_logits_dev = reduce(lambda x,y: torch.cat((x,y)), model.all_logits[::])
    all_pool_dev = reduce(lambda x,y: torch.cat((x,y)), model.all_pool[::])
    dev_labels = reduce(lambda x,y: torch.cat((x,y)), dev_labels[::])
    #save_logits(all_logits_out, 'all_ood_logits.pt')
    save_logits(all_pool_dev, '/content/drive/MyDrive/Masterarbeit/Results/1305_dev_treshold_pool.pt')
    print(all_pool_dev.size())
    model.all_logits = []
    model.all_pool = []



    #2. Treshold + Scores
    #thresholds = Tresholds()
    #scores_dev = Scores(thresholds, all_logits_dev, all_logits_out, all_pool_dev, all_pool_out, all_logits_train, all_pool_train, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    #scores_dev.calculate_scores(best_temp)
    #thresholds.calculate_tresholds(scores_dev)

    #scores = Scores(thresholds, all_logits_in, all_logits_out, all_pool_in, all_pool_out, all_logits_train, all_pool_train, model.norm_bank, model.all_classes, train_labels, dev_labels, model.class_mean, model.class_var)
    #scores.calculate_scores(best_temp)
    #scores.apply_tresholds()
   

    #3. Metriken anwenden
    #evaluate_test(args, scores)

   


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


                #alles bis auf maha und adb sollten hier möglch sein
                probs_ = torch.softmax(logits, dim=1)

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

    #for e in tqdm(test_id, desc = 'Intent examples'):
    for e in test_id:
        pred, conf, matched_example = predict_intent(e.text)
        print("-----------------")
        print(e.text)
        print(e.label)
        print(pred)
        print(conf)
        print(matched_example)
        print("-----------------")

    for e in test_ood:
        pred, conf, matched_example = predict_intent(e.text)
        print("-----------------")
        print(e.text)
        print(e.label)
        print(pred)
        print(conf)
        print(matched_example)
        print("-----------------")
    









#################################################### METHODEN ####################################################



def get_model_prediction(logits):
    #return label_pred des Models
    pred, label_pred = logits.max(dim = 1)
    return pred.cpu().detach().numpy(), label_pred.cpu().detach().numpy()

def get_softmax_score(logits):
    # Nach ID 2 (=Maxprob)
    # Input: logits eines batches
    # Return :
    # - softmax_score(Scores der prediction für alle Klassen -> max(-1) gibt den Max Wert aus)
    # - max_indices = Klasse, die den Max Wert hat 
    softmax, idx = F.softmax(logits, dim=-1).max(-1)
    return softmax.cpu().detach().numpy(), idx.cpu().detach().numpy()


def get_softmax_score_with_temp(logits, temp):
    # Nach Odin: https://openreview.net/pdf?id=H1VGkIxRZ
    # Input: 
    # - logits eines batches
    # - temp (low temp (below 1) -> makes the model more confident, high temperature (above 1) makes the model less confident)

    # Return :
    # - softmax_score(Scores der prediction für alle Klassen -> max(-1) gibt den Max Wert aus)
    # - max_indices = Klasse, die den Max Wert hat
    softmax, idx = F.softmax((logits / temp), dim=-1).max(-1)
    return softmax.cpu().detach().numpy(), idx.cpu().detach().numpy()

def get_entropy_score(logits):

    #Nach ID 11 (Gold)
    # Input: 
    # - logits eines batches
    # Return :
    # - entropy score

    softmax = F.softmax(logits, dim=-1)
    expo = torch.exp(softmax)
    expo = expo.cpu()
    return entropy(expo, axis=1)

def get_cosine_score(pooled, norm_bank):
    #von ID 2
    #hier fehlt noch norm_bank
    norm_pooled = F.normalize(pooled, dim=-1)
    cosine_score = norm_pooled @ norm_bank.t()
    cosine_score = cosine_score.max(-1)[0]
    return cosine_score.cpu().detach().numpy()


def get_energy_score(logits):
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
    maha_score = maha_score_full.min(-1)[0]
    #maha_score = -maha_score
    maha_score = maha_score.cpu().detach().numpy()
    return maha_score
    #https://www.statology.org/mahalanobis-distance-python/
    # aus distanz ein p-Value berechnen -> outlier



def get_gda_score(train_logits, test_logits_in, test_logits_out, train_labels, distance_type="euclidean"):

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
        result = result.min(axis=1)

        return result

  #= Maha oder Eucliien mit anderer Methode
  #ID 3+4 machen es so mit LinearDiscriminantAnalysis: https://github.com/pris-nlp/Generative_distance-based_OOD/blob/main/experiment.py#L248

    prob_train = F.softmax(train_logits, dim=-1)
    prob_train = prob_train.cpu().detach().numpy()

    prob_test_in = F.softmax(test_logits_in, dim=-1)
    prob_test_in = prob_test_in.cpu().detach().numpy()
    prob_test_out = F.softmax(test_logits_out, dim=-1)
    prob_test_out = prob_test_out.cpu().detach().numpy()
    #solver {‘svd’, ‘lsqr’, ‘eigen’}
    gda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, store_covariance=True)
    train_labels.cpu().detach().numpy()
    gda.fit(prob_train, train_labels)
    y_pred_in = gda.predict(prob_test_in)
    y_pred_out = gda.predict(prob_test_out)

    means =  gda.means_
    #distance_type  = "mahalanobis"
    #distance_type  = "euclidean"
    cov = gda.covariance_

    results_in = gda_help(prob_test_in, means, distance_type, cov)
    results_out = gda_help(prob_test_out, means, distance_type, cov)

    return results_in, results_out


def get_lof_score(logits_in, logits_out, pooled_in, pooled_out, train_pooled):
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
  lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination = 0.5, n_jobs=-1)
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
  

  #ID 11 macht es über cluster (centroids) und daraus dann den Abstand berechnen
  # gleiche Cluster wie für MAha?


def get_adb_score(pooled, centroids, delta):

    logits_adb = euclidean_metric(pooled, centroids)
    probs, preds = F.softmax(logits_adb.detach(), dim = 1).max(dim = 1)
    preds_ones = torch.ones_like(preds)
    preds.to('cuda:0')
    euc_dis = torch.norm(pooled - centroids[preds], 2, 1).view(-1)           
    preds_ones[euc_dis >= delta[preds]] = 0



def get_doc_score(train_logits, train_labels, logits_in, all_classes):
    #bzw DOC ID 12/13

    logits_in = logits_in.cpu().detach().numpy()

    def mu_fit(prob_pos_X):
        prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std

    mu_stds = []
    for i in range(len(all_classes)):
          pos_mu, pos_std = mu_fit(train_logits[train_labels == i, i])
          mu_stds.append([pos_mu, pos_std])
    
    thresholds = {}
    for col in range(len(all_classes)):
        threshold = max(0.5, 1 - 3 * mu_stds[col][1])
        label = all_classes[col]
        thresholds[label] = threshold
    thresholds = np.array(thresholds)
    
    y_pred = []
    for p in logits_in:
        max_class = np.argmax(p)
        max_value = np.max(p)
        threshold = max(0.5, 1 - 3 * mu_stds[max_class][1])

        if max_value > threshold:
            #y_pred.append(max_class)
            y_pred.append(1)
        else:
            #y_pred.append(data.unseen_label_id)
            y_pred.append(0)

    return np.array(y_pred)



#Reihenfolge




class Scores():

    def __init__(self, tresholds, logits_in, logits_out, pooled_in, pooled_out, logits_train, pooled_train, norm_bank, all_classes, train_labels, dev_labels, class_mean, class_var):

        #Logits & Pooled

        self.logits_in = logits_in
        self.logits_out = logits_out
        self.pooled_in = pooled_in
        self.pooled_out = pooled_out
        self.logits_train = logits_train
        self.pooled_train = pooled_train


        #sonstiges
        
        self.norm_bank = norm_bank
        self.all_classes = all_classes # all_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.train_labels = train_labels # labels aller Trainingsdaten: [1, 11, 3, 1, 14, ...]
        self.dev_labels = dev_labels
        self.class_mean = class_mean
        self.class_var = class_var

        #Scores
        
        self.logits_score_in = 0
        self.logits_score_out = 0

        self.softmax_score_in = 0
        self.softmax_score_out = 0

        self.softmax_score_temp_in = 0
        self.softmax_score_temp_out = 0

        self.cosine_score_in = 0
        self.cosine_score_out = 0

        self.energy_score_in = 0
        self.energy_score_out = 0

        self.entropy_score_in = 0
        self.entropy_score_out = 0

        self.doc_score_in = 0
        self.doc_score_out = 0

        self.gda_score_in = 0
        self.gda_score_out = 0

        self.maha_score_in = 0
        self.maha_score_out = 0
        self.maha_score_train = 0

        self.lof_score_in = 0
        self.lof_score_out = 0

        self.tresholds = tresholds

    def calculate_scores(self, best_temp):
        
        ################## Nur Logits #####################
        self.logits_score_in, _ = get_model_prediction(self.logits_in)
        self.logits_score_out, _ = get_model_prediction(self.logits_out)

        ################## SOFTMAX ########################
        self.softmax_score_in, idx_in = get_softmax_score(self.logits_in)
        self.softmax_score_out, idx_out= get_softmax_score(self.logits_out)

        ################# SOFTMAX TMP #####################
        self.softmax_score_temp_in, _ = get_softmax_score_with_temp(self.logits_in, best_temp)
        self.softmax_score_temp_out, _ = get_softmax_score_with_temp(self.logits_out, best_temp)

        ################# COSINE ##########################
        self.cosine_score_in = get_cosine_score(self.pooled_in, self.norm_bank)
        self.cosine_score_out = get_cosine_score(self.pooled_out, self.norm_bank)

        ################### ENERGY #######################
        self.energy_score_in = get_energy_score(self.logits_in)
        self.energy_score_out = get_energy_score(self.logits_out)

        ################## ENTROPY ########################
        self.entropy_score_in = get_entropy_score(self.logits_in)
        self.entropy_score_out = get_entropy_score(self.logits_out)

        #################### DOC ##########################
        # (return 0/1 -> kein treshold notwendig)
        #self.doc_score_in = get_doc_score(self.logits_train, self.train_labels, self.logits_in, self.all_classes)
        #self.doc_score_out = get_doc_score(self.logits_train, self.train_labels, self.logits_out, self.all_classes)

        ################### GDA ############################
        # ist maha und euclid
        self.gda_score_in , self.gda_score_out = get_gda_score(self.logits_train, self.logits_in, self.logits_out, self.train_labels)

        
        ################### MAHA ############################
        full_scores = True
        self.maha_score_in = get_maha_score(self.pooled_in, self.all_classes, self.class_mean, self.class_var, full_scores=full_scores)
        self.maha_score_out = get_maha_score(self.pooled_out, self.all_classes, self.class_mean, self.class_var, full_scores=full_scores)
        self.maha_score_train = get_maha_score(self.pooled_train, self.all_classes, self.class_mean, self.class_var, full_scores=full_scores)  

        
        maha_score_in = self.maha_score_in.cpu().detach().numpy()
        var = []
        for e in maha_score_in:
          var.append(np.var(e))
        print("##########")
        print(var)
        np.save("/content/drive/MyDrive/Masterarbeit/Results/full_in_maha.npy", maha_score_in)
        
        maha_score_out = self.maha_score_out.cpu().detach().numpy()
        var = []
        for e in maha_score_out:
          var.append(np.var(e))
        print("##########")
        print(var)
        np.save("/content/drive/MyDrive/Masterarbeit/Results/full_out_maha.npy", maha_score_out)

        raise NotImplementedError

        if full_scores == True:
            #ID 1: zusätzlich svm
            candidate_list = [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.2, 0.5]
            nuu = candidate_list[2]
            c_lr = svm.OneClassSVM(nu=nuu, kernel='linear', degree=2)
            #brauchen noch einen Maha Score vom Training -> oder dev??? HR???, da es im paper 1 kein Dev datensatz gibt, haben die Training genommen
            maha_train = self.maha_score_train.cpu().detach().numpy()
            c_lr.fit(maha_train)
            self.maha_score_in =  c_lr.score_samples(self.maha_score_in.cpu().detach().numpy())
            self.maha_score_out =  c_lr.score_samples(self.maha_score_out.cpu().detach().numpy())


        #################### LOF ############################
        #(return 0/1 -> kein treshold notwendig)
        # hier nochmal prüfen ob das passt? HR???

        self.lof_score_in, self.lof_score_out = get_lof_score(self.logits_in, self.logits_out, self.pooled_in, self.pooled_out, self.pooled_train)


    def apply_tresholds(self):

        ################## LOGITS ########################
        self.logits_score_in = np.where(self.logits_score_in >= self.tresholds.logits_t, 1, 0)
        self.logits_score_out = np.where(self.logits_score_out >= self.tresholds.logits_t, 1, 0)

        ################## SOFTMAX ########################
        self.softmax_score_in = np.where(self.softmax_score_in >= self.tresholds.softmax_t, 1, 0)
        self.softmax_score_out = np.where(self.softmax_score_out >= self.tresholds.softmax_t, 1, 0)
    
  
        ################## SOFTMAX TEMP ###########################
        self.softmax_score_temp_in = np.where(self.softmax_score_temp_in >= self.tresholds.sofmtax_temp_t, 1, 0)
        self.softmax_score_temp_out = np.where(self.softmax_score_temp_out >= self.tresholds.sofmtax_temp_t, 1, 0)

        ################### MAHA ############################
        self.maha_score_in = np.where(self.maha_score_in <= self.tresholds.maha_t, 1, 0)
        self.maha_score_out = np.where(self.maha_score_out <= self.tresholds.maha_t, 1, 0)

        ################## Entropy ###############
        self.entropy_score_in = np.where(self.entropy_score_in <= self.tresholds.entropy_t, 1, 0)
        self.entropy_score_out = np.where(self.entropy_score_out <= self.tresholds.entropy_t, 1, 0)

        ################## Cosine ###############
        self.cosine_score_in = np.where(self.cosine_score_in >= self.tresholds.cosine_t, 1, 0)
        self.cosine_score_out = np.where(self.cosine_score_out >= self.tresholds.cosine_t, 1, 0)
        print(self.cosine_score_in)
        print(self.cosine_score_out)
        print("#############")

        ################## Energy ###############
        self.energy_score_in = np.where(self.energy_score_in >= self.tresholds.energy_t, 1, 0)
        self.energy_score_out = np.where(self.energy_score_out >= self.tresholds.energy_t, 1, 0)

        ################### GDA ###############
        self.gda_score_in = np.where(self.gda_score_in >= self.tresholds.gda_t, 1, 0)
        self.gda_score_out = np.where(self.gda_score_out >= self.tresholds.gda_t, 1, 0)


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
        self.gda_t = 0


    def treshold_picker(self, t_pred, f, t, step, min):

        # #https://www.machinelearningplus.com/statistics/mahalanobis-distance/
        # significance_level=0.01
        # self.critical_value = chi2.ppf((1-significance_level), df=t_pred.shape[1]-1)
        # print('Critical value is: ', self.critical_value)

        if min is False:
            n = int(t_pred.size/10)
            print(n)
            return np.partition(t_pred, n-1)[n-1]
            return np.amin(t_pred)
        else:
            n = int(int(t_pred.size) - int(t_pred.size/10))
            return np.partition(t_pred, n-1)[n-1]
            return np.max(t_pred)

        
        def svm_tresholds():
            #bestimmt keinen Treshold -> sofort 0/1
            #keine guten Ergebnisse -> siehe OOD-metriken
            nuu = 0.001991961961961962
            c_lr = None
            c_lr = svm.OneClassSVM(nu=nuu, kernel='linear', degree=2)
            #c_lr = svm.OneClassSVM(gamma='scale', nu=0.01) 
            c_lr.fit(x_dev)
            y_test_id = c_lr.predict(x_test_id)
            #-1 mit 0 tauschen
            y_test_id = np.where(y_test_id == -1, 0, 1)

            y_test_ood = c_lr.predict(x_test_ood)
            #-1 mit 0 tauschen
            y_test_ood = np.where(y_test_ood == -1, 0, 1)

        def linear_regression():
            #nicht möglich, da man dev-labels braucht und die immer 1 sind 
            pass
        
        #nach Acc gesamt
        #oder nach F1
        if not isinstance(t_pred, np.ndarray):
            t_pred = t_pred.cpu().detach().numpy()

        best_t = 0
        best_acc = 0
        labels_in = np.ones_like(t_pred).astype(np.int64)

        steps = np.linspace(f,t,step)
        for i in steps:

            if min is True:
                y_pred = np.where(t_pred <= i, 1, 0)
            else:
                y_pred = np.where(t_pred >= i, 1, 0)
            acc = accuracy_score(labels_in, y_pred)

            print(acc)

            if acc > best_acc:
                best_acc = acc
                best_t = i

        return best_t

    def calculate_tresholds(self, scores):

        ################## LOGITS ########################
        self.logits_t = self.treshold_picker(scores.logits_score_in, 0, 5, 500, min=False)

        ################## SOFTMAX ########################
        self.softmax_t = self.treshold_picker(scores.softmax_score_in, 0, 1, 50, min=False)
    
        ################## SOFTMAX TEMP ###########################
        self.sofmtax_temp_t = self.treshold_picker(scores.softmax_score_temp_in, 0, 1, 50, min=False)

        ################### MAHA ############################
        self.maha_t = self.treshold_picker(scores.maha_score_in, 0, 1000, 50, min=True)

        ################## Entropy ###############
        self.entropy_t = self.treshold_picker(scores.entropy_score_in, 2.5, 3, 500, min=True)

        ################## Cosine ###############
        self.cosine_t = self.treshold_picker(scores.cosine_score_in, 0, 1, 50, min=False)
        print(self.cosine_t)
        ################## Energy ###############
        self.energy_t = self.treshold_picker(scores.energy_score_in, 0, 5, 500, min=False)
        
        ################### GDA ###############
        self.gda_t = self.treshold_picker(scores.gda_score_in, 0, 5, 500, min=False)
