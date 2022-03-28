import torch
import numpy as np
from evaluation import get_auroc, get_fpr_95
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt

def merge_keys(l, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = []
        for i in l:
            new_dict[key] += i[key]
    return new_dict

def detect_ood(args, model, prepare_dataset, test_id_dataset, test_ood_dataset, tag="test", centroids=None, delta=None):
    
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
    for batch in tqdm(test_id_dataset):

        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids=centroids, delta=delta)
            in_scores.append(ood_keys)
    in_scores = merge_keys(in_scores, keys)

    out_scores = []
    for batch in tqdm(test_ood_dataset):
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.no_grad():
            ood_keys = model.compute_ood(**batch, centroids=centroids, delta=delta)
            out_scores.append(ood_keys)
    out_scores = merge_keys(out_scores, keys)

    outputs = {}
    for key in keys:

        ins = np.array(in_scores[key], dtype=np.float64)
        outs = np.array(out_scores[key], dtype=np.float64)
        inl = np.ones_like(ins).astype(np.int64)
        outl = np.zeros_like(outs).astype(np.int64)
        scores = np.concatenate([ins, outs], axis=0)
        labels = np.concatenate([inl, outl], axis=0)

        auroc, fpr_95 = get_auroc(labels, scores), get_fpr_95(labels, scores)

        outputs[tag + "_" + key + "_auroc"] = auroc
        outputs[tag + "_" + key + "_fpr95"] = fpr_95

    wandb.log(outputs) if args.wandb == "log" else None


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

    
    labels = ['ood', 'nutrition_info', 'account_blocked', 'oil_change_how', 'time', 'weather', 'redeem_rewards', 'interest_rate', 'gas_type', 'accept_reservations', 'smart_home', 'user_name', 'report_lost_card', 'repeat', 'whisper_mode', 'what_are_your_hobbies', 'order', 'jump_start', 'schedule_meeting', 'meeting_schedule', 'freeze_account', 'what_song', 'meaning_of_life', 'restaurant_reservation', 'traffic', 'make_call', 'text', 'bill_balance', 'improve_credit_score', 'change_language', 'no', 'measurement_conversion', 'timer', 'flip_coin', 'do_you_have_pets', 'balance', 'tell_joke', 'last_maintenance', 'exchange_rate', 'uber', 'car_rental', 'credit_limit', 'restaurant_reviews', 'shopping_list', 'expiration_date', 'routing', 'meal_suggestion', 'tire_change', 'todo_list', 'card_declined', 'rewards_balance', 'change_accent', 'vaccines', 'reminder_update', 'food_last', 'change_ai_name', 'bill_due', 'who_do_you_work_for', 'share_location', 'international_visa', 'calendar', 'translate', 'carry_on', 'book_flight', 'insurance_change', 'todo_list_update', 'timezone', 'cancel_reservation', 'transactions', 'credit_score', 'report_fraud', 'spending_history', 'directions', 'spelling', 'insurance', 'what_is_your_name', 'reminder', 'where_are_you_from', 'distance', 'payday', 'flight_status', 'find_phone', 'greeting', 'alarm', 'order_status', 'confirm_reservation', 'cook_time', 'damaged_card', 'reset_settings', 'pin_change', 'replacement_card_duration', 'new_card', 'roll_dice', 'income', 'taxes', 'date', 'who_made_you', 'pto_request', 'tire_pressure', 'how_old_are_you', 'rollover_401k', 'pto_request_status', 'how_busy', 'application_status', 'recipe', 'calendar_update', 'play_music', 'yes', 'direct_deposit', 'credit_limit_change', 'gas', 'pay_bill', 'ingredients_list', 'lost_luggage', 'goodbye', 'what_can_i_ask_you', 'book_hotel', 'are_you_a_bot', 'next_song', 'change_speed', 'plug_type', 'maybe', 'w2', 'oil_change_when', 'thank_you', 'shopping_list_update', 'pto_balance', 'order_checks', 'travel_alert', 'fun_fact', 'sync_device', 'schedule_maintenance', 'apr', 'transfer', 'ingredient_substitution', 'calories', 'current_location', 'international_fees', 'calculator', 'definition', 'next_holiday', 'update_playlist', 'mpg', 'min_payment', 'change_user_name', 'restaurant_suggestion', 'travel_notification', 'cancel', 'pto_used', 'travel_suggestion', 'change_volume']
    labels_sorted = sorted(labels)
    labels = list(range(0, 152))

    cm = confusion_matrix(y_true, y_pred)

    # save numpy array as csv file
    savetxt('confusion_matrix.csv', cm, delimiter=',')

    #Classfication Report
    print(classification_report(y_true, y_pred, labels=labels))