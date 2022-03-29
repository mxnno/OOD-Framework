import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

task_to_labels = {
    'full': 151,
    'unlabeled': 2,
    'zero': 2,
    'banking': 16,
    'travel': 16,
}
def get_num_labels(args):
    if args.ood_data == 'zero':
        return task_to_labels[args.id_data] - 1 
    else:
        return task_to_labels[args.id_data]

def get_labels(id_task):

    if id_task == 'full':
        labels_name = ['ood', 'nutrition_info', 'account_blocked', 'oil_change_how', 'time', 'weather', 'redeem_rewards', 'interest_rate', 'gas_type', 'accept_reservations', 'smart_home', 'user_name', 'report_lost_card', 'repeat', 'whisper_mode', 'what_are_your_hobbies', 'order', 'jump_start', 'schedule_meeting', 'meeting_schedule', 'freeze_account', 'what_song', 'meaning_of_life', 'restaurant_reservation', 'traffic', 'make_call', 'text', 'bill_balance', 'improve_credit_score', 'change_language', 'no', 'measurement_conversion', 'timer', 'flip_coin', 'do_you_have_pets', 'balance', 'tell_joke', 'last_maintenance', 'exchange_rate', 'uber', 'car_rental', 'credit_limit', 'restaurant_reviews', 'shopping_list', 'expiration_date', 'routing', 'meal_suggestion', 'tire_change', 'todo_list', 'card_declined', 'rewards_balance', 'change_accent', 'vaccines', 'reminder_update', 'food_last', 'change_ai_name', 'bill_due', 'who_do_you_work_for', 'share_location', 'international_visa', 'calendar', 'translate', 'carry_on', 'book_flight', 'insurance_change', 'todo_list_update', 'timezone', 'cancel_reservation', 'transactions', 'credit_score', 'report_fraud', 'spending_history', 'directions', 'spelling', 'insurance', 'what_is_your_name', 'reminder', 'where_are_you_from', 'distance', 'payday', 'flight_status', 'find_phone', 'greeting', 'alarm', 'order_status', 'confirm_reservation', 'cook_time', 'damaged_card', 'reset_settings', 'pin_change', 'replacement_card_duration', 'new_card', 'roll_dice', 'income', 'taxes', 'date', 'who_made_you', 'pto_request', 'tire_pressure', 'how_old_are_you', 'rollover_401k', 'pto_request_status', 'how_busy', 'application_status', 'recipe', 'calendar_update', 'play_music', 'yes', 'direct_deposit', 'credit_limit_change', 'gas', 'pay_bill', 'ingredients_list', 'lost_luggage', 'goodbye', 'what_can_i_ask_you', 'book_hotel', 'are_you_a_bot', 'next_song', 'change_speed', 'plug_type', 'maybe', 'w2', 'oil_change_when', 'thank_you', 'shopping_list_update', 'pto_balance', 'order_checks', 'travel_alert', 'fun_fact', 'sync_device', 'schedule_maintenance', 'apr', 'transfer', 'ingredient_substitution', 'calories', 'current_location', 'international_fees', 'calculator', 'definition', 'next_holiday', 'update_playlist', 'mpg', 'min_payment', 'change_user_name', 'restaurant_suggestion', 'travel_notification', 'cancel', 'pto_used', 'travel_suggestion', 'change_volume']
        labels_id = [*range(0,151)]
    elif id_task == 'unlabeled':
        labels_name = ['ood', 'id']
        labels_id = [0, 1]
    elif id_task == 'travel':
        labels_name = ['ood', 'exchange_rate', 'car_rental', 'vaccines', 'international_visa', 'translate', 'carry_on', 'book_flight', 'timezone', 'flight_status', 'lost_luggage', 'book_hotel', 'plug_type', 'travel_alert', 'travel_notification', 'travel_suggestion']
        labels_id = [0, 38, 40, 52, 59, 61, 62, 63, 66, 80, 113, 116, 120, 128, 146, 149]
    elif id_task == 'banking':
        labels_name = ['ood', 'account_blocked', 'interest_rate', 'freeze_account',  'bill_balance', 'balance', 'routing', 'bill_due', 'transactions', 'report_fraud', 'spending_history', 'pin_change', 'pay_bill', 'order_checks', 'transfer', 'min_payment']
        labels_id = [0, 2, 7, 20, 27, 35, 45, 56, 68, 70, 71, 89, 111, 127, 133, 143]

    return  labels_name, labels_id


def save_model(model, args):

    if args.save_path == "drive":
        path = get_save_path(args)
    else:
        path = args.save_path
    model.save_pretrained(path)
    print("Model saved at: " + path)

def get_save_path(args):
    return '/content/drive/MyDrive/Masterarbeit/Trainierte_Modelle/{}/{}_{}_{}_{}_{}'.format(args.model_ID, args.id_data, args.ood_data, args.few_shot, int(args.num_train_epochs), args.tpu)

def save_tensor(tensor, path, tesnor_name="Tensor"):
    torch.save(tensor, path)
    print(tesnor_name + " saved at: " + path)





# def collate_fn(batch):
#     max_len = max([len(f["input_ids"]) for f in batch])
#     input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
#     input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
#     labels = [f["labels"] for f in batch]
#     input_ids = torch.tensor(input_ids, dtype=torch.long)
#     input_mask = torch.tensor(input_mask, dtype=torch.float)
#     labels = torch.tensor(labels, dtype=torch.long)
#     outputs = {
#         "input_ids": input_ids,
#         "attention_mask": input_mask,
#         "labels": labels,
#     }
#     return outputs
