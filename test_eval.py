import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score


def get_trehsold_with_ood(logits_in, logits_out, step, min):
    
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

def get_trehsold(logits_in, logits_out, step, min):
    
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


np.random.seed(111)
unlabeled_id = np.random.dirichlet(np.ones(2),size=50)
labeled_id = np.random.dirichlet(np.ones(16),size=50)
counter = 0
for e in unlabeled_id:
    if e[0] > e[1]:
        counter += 1
        if counter ==2:
            counter = 0
            help = e[0]
            e[0] = e[1]
            e[1] = help
         
labeled_id[0][0] = 0.52
labeled_id[0][1] = 0.48
    
np.random.seed(111)
unlabeled_ood = np.random.dirichlet(np.ones(2),size=50)
labeled_ood = np.random.dirichlet(np.ones(16),size=50)

counter = 0
for e in unlabeled_ood:
    if e[0] < e[1]:
        counter += 1
        if counter ==2:
            counter = 0
            help = e[1]
            e[1] = e[0]
            e[0] = help

counter = 0
for e in labeled_ood:
    idx = e.argmax()
    if idx != 0:
        counter += 1
        if counter ==2:
            counter = 0
            help = e[0]
            e[0] = e[idx]
            e[idx] = help

#########################################################################
#1. 0 > 1 bzw. 0 > 1-15
#labeled:
idx_la_in = labeled_id.argmax(axis = -1)
idx_la_out = labeled_ood.argmax(axis = -1)

pred_la_in = np.where(idx_la_in > 0, 1, 0)
pred_la_out = np.where(idx_la_out > 0, 1, 0)

#unlabeled
idx_un_in = unlabeled_id.argmax(axis = -1)
idx_un_out = unlabeled_ood.argmax(axis = -1)

pred_un_in = np.where(idx_un_in > 0, 1, 0)
pred_un_out = np.where(idx_un_out > 0, 1, 0)


#Validation
labels_in = np.ones_like(pred_la_in).astype(np.int64)
labels_out = np.zeros_like(pred_la_out).astype(np.int64)


#labeled:
in_acc = accuracy_score(labels_in, pred_la_in)
out_acc = accuracy_score(labels_out, pred_la_out)
print("1 | labeled : " + str(in_acc + out_acc))

#labeled:
in_acc = accuracy_score(labels_in, pred_un_in)
out_acc = accuracy_score(labels_out, pred_un_out)
print(in_acc)
print(out_acc)
print("1 | unlabeled : " + str(in_acc + out_acc))
#########################################################################
#########################################################################
#2. treshold OOD
t_la = get_trehsold_with_ood(labeled_id, labeled_ood, 500, False)
t_un = get_trehsold_with_ood(unlabeled_id, unlabeled_ood, 500, False)
print(t_un)
#labeled:
pred_la_in =  np.where(labeled_id[:,0] > t_la, 0, 1)
pred_la_out =  np.where(labeled_ood[:,0] > t_la, 0, 1)

#unlabeled
pred_un_in =  np.where(unlabeled_id[:,0] > t_un, 0, 1)
pred_un_out =  np.where(unlabeled_ood[:,0] > t_un, 0, 1)


#Validation
labels_in = np.ones_like(pred_la_in).astype(np.int64)
labels_out = np.zeros_like(pred_la_out).astype(np.int64)


#labeled:
in_acc = accuracy_score(labels_in, pred_la_in)
out_acc = accuracy_score(labels_out, pred_la_out)
print("2 | labeled : " + str(in_acc + out_acc))
print(in_acc)
print(out_acc)
#labeled:
in_acc = accuracy_score(labels_in, pred_un_in)
out_acc = accuracy_score(labels_out, pred_un_out)
print("2 | unlabeled : " + str(in_acc + out_acc))
#########################################################################
#########################################################################
#3. treshold ID

t_la = get_trehsold(labeled_id, labeled_ood, 500, False)
t_un = get_trehsold(unlabeled_id, unlabeled_ood, 500, False)

print(t_un)
#labeled:
pred_la_in =  np.where(labeled_id[:,1:].max(axis = 1) > t_la, 1, 0)
pred_la_out =  np.where(labeled_ood[:,1:].max(axis = 1) > t_la, 1, 0)

#unlabeled
pred_un_in =  np.where(unlabeled_id[:,1:].max(axis = 1) > t_un, 1, 0)
pred_un_out =  np.where(unlabeled_ood[:,1:].max(axis = 1) > t_un, 1, 0)


#Validation
labels_in = np.ones_like(pred_la_in).astype(np.int64)
labels_out = np.zeros_like(pred_la_out).astype(np.int64)


#labeled:
in_acc = accuracy_score(labels_in, pred_la_in)
out_acc = accuracy_score(labels_out, pred_la_out)
#print("3 | labeled : " + str(in_acc + out_acc))

#labeled:
in_acc = accuracy_score(labels_in, pred_un_in)
out_acc = accuracy_score(labels_out, pred_un_out)
print("3 | unlabeled : " + str(in_acc + out_acc))
print(in_acc)
print(out_acc)
#########################################################################

