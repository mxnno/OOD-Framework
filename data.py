import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
import re

datasets.logging.set_verbosity(datasets.logging.ERROR)


def preprocess_data(dataset_name, args, num_labels, tokenizer, no_Dataloader=False, model_type="SequenceClassification"):

    print("Dataset: " + dataset_name)
    if dataset_name == 'clinc150':
        raw_datasets = load_clinc(args.few_shot, num_labels, args.ood_data)
    elif dataset_name == 'clinc150_AUG':
        raw_datasets = load_clinc_with_Augmentation(args.few_shot, num_labels, args.ood_data)
    elif dataset_name == 'test_eval':
        raw_datasets = test_evaluation_dataset()
    else:
        print ("Nicht gefunden: " + dataset_name)
        raise NotImplementedError


    def tokenize_function(example):
        if model_type == 'LanguageModeling':
            result = tokenizer(example["text"])
            if tokenizer.is_fast:
                result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
            return result
        else:
            return tokenizer(example["text"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    

    if model_type == 'LanguageModeling':
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,  mlm_probability=0.15)
    elif model_type == 'SequenceClassification':
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length")
    else:
        raise NotImplementedError
        

    #Columns anpassen
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("intent", "labels")
    if model_type == 'LanguageModeling':
        tokenized_datasets = tokenized_datasets.remove_columns(["labels"])
    tokenized_datasets.set_format("torch")

    if no_Dataloader:
        return tokenized_datasets["train"], tokenized_datasets["validation"], data_collator

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator
    )
    test__ood_dataloader = DataLoader(
        tokenized_datasets["test_ood"], batch_size=args.batch_size, collate_fn=data_collator
    )
    test_id_dataloader = DataLoader(
        tokenized_datasets["test_id"], batch_size=args.batch_size, collate_fn=data_collator
    )

    #for batch in eval_dataloader:
        #print({k: v.shape for k, v in batch.items()})

    return train_dataloader, eval_dataloader, test_id_dataloader, test__ood_dataloader


def load_clinc(few_shot, num_labels, ood_data):
    #few_shot: Anteil vom ursprünglichen Datensatz
    #num_labels: Anzahl label -> 2 oder 151
    #ood_data: 'zero' -> nur ID, sonst ID + OOD

    num_shards = int(100/few_shot)

    if num_labels == 2:
        names=['ood', 'id']
    else:
        names=['ood', 'nutrition_info', 'account_blocked', 'oil_change_how', 'time', 'weather', 'redeem_rewards', 'interest_rate', 'gas_type', 'accept_reservations', 'smart_home', 'user_name', 'report_lost_card', 'repeat', 'whisper_mode', 'what_are_your_hobbies', 'order', 'jump_start', 'schedule_meeting', 'meeting_schedule', 'freeze_account', 'what_song', 'meaning_of_life', 'restaurant_reservation', 'traffic', 'make_call', 'text', 'bill_balance', 'improve_credit_score', 'change_language', 'no', 'measurement_conversion', 'timer', 'flip_coin', 'do_you_have_pets', 'balance', 'tell_joke', 'last_maintenance', 'exchange_rate', 'uber', 'car_rental', 'credit_limit', 'restaurant_reviews', 'shopping_list', 'expiration_date', 'routing', 'meal_suggestion', 'tire_change', 'todo_list', 'card_declined', 'rewards_balance', 'change_accent', 'vaccines', 'reminder_update', 'food_last', 'change_ai_name', 'bill_due', 'who_do_you_work_for', 'share_location', 'international_visa', 'calendar', 'translate', 'carry_on', 'book_flight', 'insurance_change', 'todo_list_update', 'timezone', 'cancel_reservation', 'transactions', 'credit_score', 'report_fraud', 'spending_history', 'directions', 'spelling', 'insurance', 'what_is_your_name', 'reminder', 'where_are_you_from', 'distance', 'payday', 'flight_status', 'find_phone', 'greeting', 'alarm', 'order_status', 'confirm_reservation', 'cook_time', 'damaged_card', 'reset_settings', 'pin_change', 'replacement_card_duration', 'new_card', 'roll_dice', 'income', 'taxes', 'date', 'who_made_you', 'pto_request', 'tire_pressure', 'how_old_are_you', 'rollover_401k', 'pto_request_status', 'how_busy', 'application_status', 'recipe', 'calendar_update', 'play_music', 'yes', 'direct_deposit', 'credit_limit_change', 'gas', 'pay_bill', 'ingredients_list', 'lost_luggage', 'goodbye', 'what_can_i_ask_you', 'book_hotel', 'are_you_a_bot', 'next_song', 'change_speed', 'plug_type', 'maybe', 'w2', 'oil_change_when', 'thank_you', 'shopping_list_update', 'pto_balance', 'order_checks', 'travel_alert', 'fun_fact', 'sync_device', 'schedule_maintenance', 'apr', 'transfer', 'ingredient_substitution', 'calories', 'current_location', 'international_fees', 'calculator', 'definition', 'next_holiday', 'update_playlist', 'mpg', 'min_payment', 'change_user_name', 'restaurant_suggestion', 'travel_notification', 'cancel', 'pto_used', 'travel_suggestion', 'change_volume']
    
    classlabel = ClassLabel(num_classes = num_labels, names=names)

    datasets_dict = load_dataset("clinc_oos", "small")

    def change_ood_label(example):
        #OOD = label 0

        if example['intent'] == 0:
            example['intent'] = 42
        elif example['intent'] == 42:
            example['intent'] = 0

        return example

    #Trainingsdaten ID/OOD aufteilen + verkleinern
    dataset = datasets_dict['train']
    dataset = dataset.map(change_ood_label)
    ood = dataset.filter(lambda example: example['intent'] == 0)
    id = dataset.filter(lambda example: example['intent'] != 0)

    #Train Dataset zufällig shuffeln und reduzieren
    shuffled_train = id.shuffle(seed=42)
    sorted_train = shuffled_train.sort('intent')
    sharded_train = sorted_train.shard(num_shards=num_shards, index=0)
    
    if ood_data == 'zero':
        train_dataset = sharded_train
    else:
        train_dataset = concatenate_datasets([sharded_train, ood])

    train_dataset = train_dataset.cast_column("intent", classlabel)

    #Validation Daten ID/OOD aufteilen
    val_dataset = datasets_dict['validation']
    val_dataset = val_dataset.map(change_ood_label)
    if ood_data == 'zero':
        val_dataset = val_dataset.filter(lambda example: example['intent'] != 0)
    val_dataset = val_dataset.cast_column("intent", classlabel)
    
    #Testdaten ID/OOD aufteilen
    test_dataset = datasets_dict['test']
    test_dataset = test_dataset.map(change_ood_label)
    test_ood_dataset = test_dataset.filter(lambda example: example['intent'] == 0)
    test_id_dataset = test_dataset.filter(lambda example: example['intent'] != 0)
    test_ood_dataset = test_ood_dataset.cast_column("intent", classlabel)
    test_id_dataset = test_id_dataset.cast_column("intent", classlabel)

    #Falls 2 Klassen 
    def change_label_binary(example):
        if example['intent'] != 0:
            example['intent'] = 1
        return example

    if num_labels == 2:
        train_dataset = train_dataset.map(change_label_binary)
        val_dataset = val_dataset.map(change_label_binary)
        test_ood_dataset = test_ood_dataset.map(change_label_binary)
        test_id_dataset = test_id_dataset.map(change_label_binary)

    return DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test_ood': test_ood_dataset, 'test_id': test_id_dataset})

def load_clinc_with_Augmentation(few_shot, num_labels, ood_data):
    
    clinc_DatasetDict = load_clinc(few_shot, num_labels, ood_data)

    def prepare_txt(example):

        #index und /t vor dem Satz entfernen
        example['text'] = re.sub(r'^.*?/t', '', example['text'])
        #. und ? als Satzzeichen entfernen
        example['text'] = example['text'].strip(".?")
        
        # label hinzufügen (sind alle OOD)
        example['intent'] = 0

        return example

    for datafile in ['/content/OOD-Framework/data/Augmentation/wiki.txt', '/content/OOD-Framework/data/Augmentation/subset_books.txt']:


        if num_labels == 2:
            names=['ood', 'id']
        else:
            names=['ood', 'nutrition_info', 'account_blocked', 'oil_change_how', 'time', 'weather', 'redeem_rewards', 'interest_rate', 'gas_type', 'accept_reservations', 'smart_home', 'user_name', 'report_lost_card', 'repeat', 'whisper_mode', 'what_are_your_hobbies', 'order', 'jump_start', 'schedule_meeting', 'meeting_schedule', 'freeze_account', 'what_song', 'meaning_of_life', 'restaurant_reservation', 'traffic', 'make_call', 'text', 'bill_balance', 'improve_credit_score', 'change_language', 'no', 'measurement_conversion', 'timer', 'flip_coin', 'do_you_have_pets', 'balance', 'tell_joke', 'last_maintenance', 'exchange_rate', 'uber', 'car_rental', 'credit_limit', 'restaurant_reviews', 'shopping_list', 'expiration_date', 'routing', 'meal_suggestion', 'tire_change', 'todo_list', 'card_declined', 'rewards_balance', 'change_accent', 'vaccines', 'reminder_update', 'food_last', 'change_ai_name', 'bill_due', 'who_do_you_work_for', 'share_location', 'international_visa', 'calendar', 'translate', 'carry_on', 'book_flight', 'insurance_change', 'todo_list_update', 'timezone', 'cancel_reservation', 'transactions', 'credit_score', 'report_fraud', 'spending_history', 'directions', 'spelling', 'insurance', 'what_is_your_name', 'reminder', 'where_are_you_from', 'distance', 'payday', 'flight_status', 'find_phone', 'greeting', 'alarm', 'order_status', 'confirm_reservation', 'cook_time', 'damaged_card', 'reset_settings', 'pin_change', 'replacement_card_duration', 'new_card', 'roll_dice', 'income', 'taxes', 'date', 'who_made_you', 'pto_request', 'tire_pressure', 'how_old_are_you', 'rollover_401k', 'pto_request_status', 'how_busy', 'application_status', 'recipe', 'calendar_update', 'play_music', 'yes', 'direct_deposit', 'credit_limit_change', 'gas', 'pay_bill', 'ingredients_list', 'lost_luggage', 'goodbye', 'what_can_i_ask_you', 'book_hotel', 'are_you_a_bot', 'next_song', 'change_speed', 'plug_type', 'maybe', 'w2', 'oil_change_when', 'thank_you', 'shopping_list_update', 'pto_balance', 'order_checks', 'travel_alert', 'fun_fact', 'sync_device', 'schedule_maintenance', 'apr', 'transfer', 'ingredient_substitution', 'calories', 'current_location', 'international_fees', 'calculator', 'definition', 'next_holiday', 'update_playlist', 'mpg', 'min_payment', 'change_user_name', 'restaurant_suggestion', 'travel_notification', 'cancel', 'pto_used', 'travel_suggestion', 'change_volume']
    

        data_dict = load_dataset('text', data_files={'train': datafile})
        train_dataset = data_dict['train']
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.shard(num_shards=70, index=0)
        train_dataset = train_dataset.map(prepare_txt)
        classlabel = ClassLabel(num_classes = num_labels, names = names)
        train_dataset = train_dataset.cast_column("intent", classlabel)

        clinc_DatasetDict['train'] = concatenate_datasets([clinc_DatasetDict['train'], train_dataset])

    return clinc_DatasetDict
    

def test_evaluation_dataset(num_labels):

    #4 * label = 32
    #4 * label = 0

    def prepare_txt_ID(example):
        
        # label hinzufügen (sind alle ID)
        if num_labels == 2:
            example['intent'] = 1
        else:
            example['intent'] = 32

        return example

    def prepare_txt_OOD(example):
        
        # label hinzufügen (sind alle OOD)
        example['intent'] = 0

        return example

    if num_labels == 2:
        names=['ood', 'id']
    else:
        names=['ood', 'nutrition_info', 'account_blocked', 'oil_change_how', 'time', 'weather', 'redeem_rewards', 'interest_rate', 'gas_type', 'accept_reservations', 'smart_home', 'user_name', 'report_lost_card', 'repeat', 'whisper_mode', 'what_are_your_hobbies', 'order', 'jump_start', 'schedule_meeting', 'meeting_schedule', 'freeze_account', 'what_song', 'meaning_of_life', 'restaurant_reservation', 'traffic', 'make_call', 'text', 'bill_balance', 'improve_credit_score', 'change_language', 'no', 'measurement_conversion', 'timer', 'flip_coin', 'do_you_have_pets', 'balance', 'tell_joke', 'last_maintenance', 'exchange_rate', 'uber', 'car_rental', 'credit_limit', 'restaurant_reviews', 'shopping_list', 'expiration_date', 'routing', 'meal_suggestion', 'tire_change', 'todo_list', 'card_declined', 'rewards_balance', 'change_accent', 'vaccines', 'reminder_update', 'food_last', 'change_ai_name', 'bill_due', 'who_do_you_work_for', 'share_location', 'international_visa', 'calendar', 'translate', 'carry_on', 'book_flight', 'insurance_change', 'todo_list_update', 'timezone', 'cancel_reservation', 'transactions', 'credit_score', 'report_fraud', 'spending_history', 'directions', 'spelling', 'insurance', 'what_is_your_name', 'reminder', 'where_are_you_from', 'distance', 'payday', 'flight_status', 'find_phone', 'greeting', 'alarm', 'order_status', 'confirm_reservation', 'cook_time', 'damaged_card', 'reset_settings', 'pin_change', 'replacement_card_duration', 'new_card', 'roll_dice', 'income', 'taxes', 'date', 'who_made_you', 'pto_request', 'tire_pressure', 'how_old_are_you', 'rollover_401k', 'pto_request_status', 'how_busy', 'application_status', 'recipe', 'calendar_update', 'play_music', 'yes', 'direct_deposit', 'credit_limit_change', 'gas', 'pay_bill', 'ingredients_list', 'lost_luggage', 'goodbye', 'what_can_i_ask_you', 'book_hotel', 'are_you_a_bot', 'next_song', 'change_speed', 'plug_type', 'maybe', 'w2', 'oil_change_when', 'thank_you', 'shopping_list_update', 'pto_balance', 'order_checks', 'travel_alert', 'fun_fact', 'sync_device', 'schedule_maintenance', 'apr', 'transfer', 'ingredient_substitution', 'calories', 'current_location', 'international_fees', 'calculator', 'definition', 'next_holiday', 'update_playlist', 'mpg', 'min_payment', 'change_user_name', 'restaurant_suggestion', 'travel_notification', 'cancel', 'pto_used', 'travel_suggestion', 'change_volume']
    

    classlabel = ClassLabel(num_classes = num_labels, names = names)

    #ID
    datafile = '/content/OOD-Framework/data/Test_Evaluation/test_id.txt'
    data_dict_id = load_dataset('text', data_files={'test': datafile})
    test_dataset_id = data_dict_id['test']
    test_dataset_id = test_dataset_id.map(prepare_txt_ID)
    test_dataset_id = test_dataset_id.cast_column("intent", classlabel)
    test_dataset_id['intent'][0] = 0

    #ood
    datafile = '/content/OOD-Framework/data/Test_Evaluation/test_ood.txt'
    data_dict_od = load_dataset('text', data_files={'test': datafile})
    test_dataset_ood = data_dict_od['test']
    test_dataset_ood = test_dataset_ood.map(prepare_txt_OOD)
    test_dataset_ood = test_dataset_ood.cast_column("intent", classlabel)
    if num_labels == 2:
        test_dataset_ood['intent'][0] = 32
    else:
        test_dataset_ood['intent'][0] = 1
   


    full_test = concatenate_datasets([test_dataset_id, test_dataset_ood])

    return DatasetDict({'train': full_test, 'validation': full_test, 'test_ood': test_dataset_ood, 'test_id': test_dataset_id})

