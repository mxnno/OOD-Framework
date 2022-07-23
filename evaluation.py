from gc import get_threshold
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from datasets import load_metric
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils_DNNC import *
from utils.utils import get_result_path
import csv
import os
import torch
from functools import reduce
from scipy.stats import entropy
from utils.utils_ADB import euclidean_metric


def evaluate_metriken_ohne_Treshold(args, scores):

    #Alle mit _ocsvm
    #evtl noch gda
    if args.model_ID == 14:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'maha_score']
    else:
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score', 'gda_eucl_score', 'gda_maha_score', 'varianz_score']

    if args.ood_data != "zero":
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score','varianz_score']

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

            if score_name != 'varianz_score':
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

    if args.ood_data != "zero":
        score_list = ['logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'energy_score', 'entropy_score', 'maha_score', 'varianz_score']
        
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
        score_list = ['lof_score', 'doc_score', 'logits_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score', 'gda_eucl_score', 'gda_maha_score']
    
    if args.ood_data != "zero":
        score_list = ['logits_score', 'doc_score', 'softmax_score', 'softmax_score_temp', 'cosine_score', 'maha_score']

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


def evaluate_method_combination(args, combis, scores_in, scores_out, Ttype):

    if args.model_ID == 2:
        if "margin" in args.model_name_or_path:
            id = "2_m"
        else:
            id = "2_s"
    else:
        id = "3"

    header = ['Method', "in_acc + out_recall", "in_acc", "in_recall", "in_f1", "out_acc", "out_recall", "out_f1", "acc", "recall", "f1", "roc_auc", "fpr_95"]
    csvPath = "/content/drive/MyDrive/Masterarbeit/Kombination/Methoden/2/" + id + "/" + str(args.id_data) + "_" + str(args.few_shot) + "_" + str(args.seed)
    if not os.path.isdir(csvPath):
        os.mkdir(csvPath)
    csvPath += "/combi_" + Ttype + ".csv"
    with open(csvPath, 'w', encoding='utf-8') as csvf:

        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(header)

        for i, score_name in enumerate(combis):
            data = []
            y_pred_in = scores_in[i]
            y_pred_out = scores_out[i]

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

            writer.writerow([score_name,] + data)

    print("Result file saved at: " + csvPath)


def evaluate_mit_OOD(args, scores):

    score_list = ['logits_idx', 'softmax_temp_idx', 'cosine_idx', 'maha_idx']
    header = ['Method', "in_acc + out_recall", "in_acc", "in_recall", "in_f1", "out_acc", "out_recall", "out_f1", "acc", "recall", "f1", "roc_auc", "fpr_95"]
    csvPath = get_result_path(args)
    if not os.path.isdir(get_result_path(args)):
        os.mkdir(get_result_path(args))
    csvPath += "/results_mit_OOD_classic.csv"
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

def evaluate(args, model, eval_id, eval_ood, centroids=None, delta=None, tag=None):

    #Test-ID:
    for batch in tqdm(eval_id):
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
    for batch in tqdm(eval_ood):
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

    # if args.model_ID == 15 and centroids is not None:

    #     def get_adb_score(pooled, centroids, delta):

    #         logits_adb = euclidean_metric(pooled, centroids)
    #         #qwert
    #         #Kann ich die logits rausnehmen für anderen Methoden???
    #         # -> heir ja nur Softmax
    #         probs, preds = F.softmax(logits_adb.detach(), dim = 1).max(dim = 1)
            
    #         #probs, preds = logits_adb.max(dim = 1)
    #         preds_ones = torch.ones_like(preds)
    #         preds.to('cuda:0')
    #         euc_dis = torch.norm(pooled - centroids[preds], 2, 1).view(-1)           
    #         preds_ones[euc_dis >= delta[preds]] = 0
    #         return preds_ones.detach().cpu().numpy()

    #     #logits_score_in = get_adb_score(all_pool_in, centroids, delta)
    #     #logits_score_out = get_adb_score(all_pool_out, centroids, delta)
    #     #print(logits_score_in)
    #     #print(logits_score_out)

    #     #_, _, all_logits_in , all_logits_out = detect_ood_adb2(centroids, delta, all_pool_in, all_pool_out, all_pool_dev, all_pool_train)
        
    #     #Das aus dem else Fall muss auch noch mit ausgeführt werden!
    #     all_logits_in = euclidean_metric(all_pool_in, centroids)
    #     all_logits_out = euclidean_metric(all_pool_out, centroids)

    #     logits_score_in = all_logits_in.max(dim = 1)[0]
    #     logits_score_in = logits_score_in.cpu().detach().numpy()
    #     logits_score_out = all_logits_out.max(dim = 1)[0]
    #     logits_score_out = logits_score_out.cpu().detach().numpy()

    #     t = get_treshold_eval(logits_score_in, logits_score_out, np.min(logits_score_out), max(logits_score_in), 500, min=False)

    #     logits_score_in = np.where(logits_score_in >= t, 1, 0)
    #     logits_score_out = np.where(logits_score_out >= t, 1, 0)
    #     print(logits_score_in)
    #     print(logits_score_out)

    if args.ood_data == 'zero':
        print("00000")
        logits_score_in = all_logits_in.max(dim = 1)[0]
        logits_score_in = logits_score_in.cpu().detach().numpy()
        logits_score_out = all_logits_out.max(dim = 1)[0]
        logits_score_out = logits_score_out.cpu().detach().numpy()

        t = get_treshold_eval(logits_score_in, logits_score_out, np.min(logits_score_out), max(logits_score_in), 500, min=False)
        print(t)

        logits_score_in = np.where(logits_score_in >= t, 1, 0)
        logits_score_out = np.where(logits_score_out >= t, 1, 0)
        print(logits_score_in)
        print(logits_score_out)

        labels_in = np.ones_like(logits_score_in).astype(np.int64)
        labels_out = np.zeros_like(logits_score_out).astype(np.int64)

        in_acc = accuracy_score(labels_in, logits_score_in)
        out_acc = accuracy_score(labels_out, logits_score_out)

        results = {"acc": in_acc + out_acc}
        return results
    else:
        #Idee bei OOD Trainignsdaten: schauen ob in Klasse 0 oder nicht
        # idx_in= all_logits_in.max(dim = 1)[0]
        # idx_in = idx_in.cpu().detach().numpy()

        # idx_out = all_logits_out.max(dim = 1)[0]
        # idx_out = idx_out.cpu().detach().numpy()


        # logits_score_in = np.where(idx_in > 0, 1, 0)
        # logits_score_out = np.where(idx_out > 0, 1, 0)
        # print(logits_score_in)
        # print(logits_score_out)

        # labels_in = np.ones_like(logits_score_in).astype(np.int64)
        # labels_out = np.zeros_like(logits_score_out).astype(np.int64)

        # in_acc = accuracy_score(labels_in, logits_score_in)
        # out_acc = accuracy_score(labels_out, logits_score_out)

        # results = {"acc": in_acc + out_acc}
        # return results

        #logits_score_in = all_logits_in.max(dim = 1)[0]
        logits_score_in = all_logits_in.cpu().detach().numpy()
        logits_score_out = all_logits_out.cpu().detach().numpy()

        #Variante 1: 0 > 1
        idx_in = logits_score_in.argmax(axis = -1)
        idx_out = logits_score_out.argmax(axis = -1)

        print(idx_in)
        print(idx_out)

        pred_in = np.where(idx_in > 0, 1, 0)
        pred_out = np.where(idx_out > 0, 1, 0)

        labels_in = np.ones_like(pred_in).astype(np.int64)
        labels_out = np.zeros_like(pred_out).astype(np.int64)

        in_acc = accuracy_score(labels_in, pred_in)
        out_acc = accuracy_score(labels_out, pred_out)

        v1 = in_acc + out_acc
        print("V_org: " + str(v1))

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

        results = {"acc": max(v1, v2)}
        return results
    



# def evaluate(args, model, eval_dataset, tag=""):
    
#     #F1
#     metric = load_metric("accuracy")

#     def compute_metrics(preds, labels):
#         preds = np.argmax(preds, axis=1)
#         result = metric.compute(predictions=preds, references=labels, )
#         if len(result) > 1:
#             result["score"] = np.mean(list(result.values())).item()
#         return result


#     label_list, logit_list = [], []
#     for batch in tqdm(eval_dataset):
#         model.eval()
        
#         if args.model_ID == 8:
#             for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataset, desc="Evaluating"):
#                 input_ids = input_ids.to(args.device)
#                 input_mask = input_mask.to(args.device)
#                 segment_ids = segment_ids.to(args.device)

#             outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
#             labels = label_ids.numpy()
#         else:
#             labels = batch["labels"].detach().cpu().numpy()
#             batch = {key: value.to(args.device) for key, value in batch.items()}
#             batch["labels"] = None
#             outputs = model(**batch)

#         logits_2 = outputs[0]
#         logits = logits_2.detach().cpu().numpy()
#         label_list.append(labels)
#         logit_list.append(logits)


#     preds = np.concatenate(logit_list, axis=0)

#     labels = np.concatenate(label_list, axis=0)
#     results = compute_metrics(preds, labels)


#     preds1 = np.argmax(preds, axis=1)
#     acc, f1 = get_acc_and_f1(preds1, labels, model.num_labels)
#     print("acc_" + tag + ": " + str(acc))
#     print("f1_" + tag + ": " + str(f1))

#     results = {"accuracy_" + tag: acc, "f1_" + tag: f1}
#     return results

    
def evaluate_DNNC(args, model, tokenizer, train, test_id, test_ood):

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

    
    for e in tqdm(test_ood, desc = 'OOD examples'):
        

        pred, conf, matched_example = predict_intent(e.text)
        pred_ood.append(conf)


    pred_id = np.array(pred_id)
    pred_id = np.where(pred_id >= 0.5, 1, 0)

    pred_ood = np.array(pred_ood)
    pred_ood = np.where(pred_ood >= 0.5, 1, 0)

    print(pred_id)
    print(pred_ood)
    labels_in = np.ones_like(pred_id).astype(np.int64)
    labels_out = np.zeros_like(pred_ood).astype(np.int64)

    in_acc = accuracy_score(labels_in, pred_id)
    out_acc = accuracy_score(labels_out, pred_ood)

    results = {"acc": in_acc + out_acc}
    return results
    


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



def get_treshold_eval(pred_in, pred_out, f, t, step, min):
    threshold = 0
    best_acc = 0
    labels_in = np.ones_like(pred_in).astype(np.int64)
    labels_out = np.zeros_like(pred_out).astype(np.int64)

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


