from scipy.fftpack import idstn
import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
import re
import shutil
from utils.utils import get_labels, get_num_labels

datasets.logging.set_verbosity(datasets.logging.ERROR)


def preprocess_data(args, tokenizer, no_Dataloader=False, model_type="SequenceClassification"):

    print("Dataset: " + args.dataset)
    if args.dataset == 'clinc150':
        raw_datasets = load_clinc(args)
    elif args.dataset == 'clinc150_AUG':
        raw_datasets = load_clinc_with_Augmentation(args)
    elif args.dataset == 'clinc150_AUG_ID':
        raw_datasets = load_clinc_with_ID_Augmentation(args)
    elif args.dataset == 'test_eval':
        raw_datasets = test_evaluation_dataset()
    else:
        print ("Nicht gefunden: " + args.dataset)
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
    trainval_dataloader = DataLoader(
        tokenized_datasets["trainval"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )
    val_id_dataloader = DataLoader(
        tokenized_datasets["val_id"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )
    val_ood_dataloader = DataLoader(
        tokenized_datasets["val_ood"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=args.batch_size, collate_fn=data_collator
    )
    test_ood_dataloader = DataLoader(
        tokenized_datasets["test_ood"], batch_size=args.batch_size, collate_fn=data_collator
    )
    test_id_dataloader = DataLoader(
        tokenized_datasets["test_id"], batch_size=args.batch_size, collate_fn=data_collator
    )

    #for batch in eval_dataloader:
        #print({k: v.shape for k, v in batch.items()})

    return train_dataloader, trainval_dataloader, val_id_dataloader, val_ood_dataloader, test_dataloader, test_id_dataloader, test_ood_dataloader


def load_clinc(args):
    #few_shot: Anteil vom ursprünglichen Datensatz
    #num_labels: Anzahl label -> 2 oder 151
    #ood_data: 'zero' -> nur ID, sonst ID + OOD
    ood_data = args.ood_data
    num_shards = int(100/(args.few_shot*2))

    ood_original = False
    
    #ic = label_ids[1] #Das muss noch überarbeitet werdne ???

    #Label Namen + Ids
    num_labels = get_num_labels(args)
    label_names, label_ids = get_labels(args)
    print(label_names)
    print(label_ids)
    if args.model_ID == 8:
        num_labels = len(label_names)

    print(num_labels)

    if args.ood_data == 'zero':
        classlabel = ClassLabel(num_classes=num_labels, names=label_names)
    else:
        classlabel = ClassLabel(num_classes=num_labels + 1, names=['ood'] + label_names)

    def set_OOD_as_0(example):
        #OOD = label 0
        if example['intent'] == 0:
            example['intent'] = 42
        elif example['intent'] == 42:
            example['intent'] = 0
        return example

    def set_label_to_OOD(example):
        example['intent'] = 0
        return example

    def set_label_to_ID(example):


        if num_labels == 2 or num_labels == 1:
            example['intent'] = 1
        else:
            example['intent'] = label_ids.index(example['intent'])
        return example

        # #falls 3 Klassen (OOD/OOC/IC) mit 0/1/2
        # #0 bleibt 0, OOC -> 1
        # #0 ist in label:ids
        # if example['intent'] not in label_ids[1:]:
        #     example['intent'] = 0
        # elif example['intent'] in label_ids[1:] and example['intent'] != ic:
        #     example['intent'] = 1
        # else:
        #     example['intent'] = 2


    #Cache leeren
    shutil.rmtree('/root/.cache/huggingface/datasets/clinc_oos/small/')
    #Datensatz laden
    datasets_dict = load_dataset("clinc_oos", "small", keep_in_memory=False)


    ########################################################### Train ###############################################################
    #Trainingsdaten ID/OOD aufteilen + verkleinern
    train_dataset = datasets_dict['train']
    train_dataset = train_dataset.map(set_OOD_as_0)

    #ID Daten zufällig shuffeln und reduzieren auf n Few-Shot
    id = train_dataset.filter(lambda example: example['intent'] in label_ids)
    print(id)
    id = id.shuffle(seed=args.seed)
    id = id.sort('intent')
    id = id.shard(num_shards=num_shards, index=0)
    id = id.map(set_label_to_ID)

    #Falls OOD:
    if ood_data != 'zero':
        #OOD Daten zufällig shuffeln und reduzieren auf 10*n Few-Shot

        if ood_original is True:
            #wenn OOD nur original OOD
            ood = train_dataset.filter(lambda example: example['intent']==0)
        else:

            #wenn OOD alle bis auf Domain
            ood = train_dataset.filter(lambda example: example['intent'] not in label_ids)
            
            ood = ood.shuffle(seed=args.seed)
            ood = ood.sort('intent')
            ood = ood.shard(num_shards=num_shards*10, index=0)
            ood = ood.map(set_label_to_OOD)

        train_dataset = concatenate_datasets([id, ood])

    else:
        train_dataset = id
    
    #train_dataset = train_dataset.shuffle(seed=args.seed)
    train_dataset = train_dataset.cast_column("intent", classlabel)
    #train_dataset.to_csv('training.csv')


    ########################################################### Validation ###############################################################

    #Validation Daten (hälfte von Trainingsdaten)
    # - ID Daten reduzieren
    # - OOD Daten bleiben
    val_dataset = datasets_dict['validation']
    val_dataset = val_dataset.map(set_OOD_as_0)

    #ID Daten zufällig shuffeln und reduzieren auf n Few-Shot
    val_id = val_dataset.filter(lambda example: example['intent'] in label_ids)
    val_id = val_id.shuffle(seed=args.seed)
    val_id = val_id.sort('intent')
    val_id = val_id.shard(num_shards=num_shards, index=0)
    val_id = val_id.map(set_label_to_ID)
    val_id.cast_column("intent", classlabel)

    #Falls OOD:
    if ood_data != 'zero':
        #OOD Daten zufällig shuffeln und reduzieren auf 10*n Few-Shot

        #OOD Daten zufällig shuffeln und reduzieren auf 3*n Few-Shot
        if ood_original is True:
            #wenn OOD nur original OOD
            val_ood = val_dataset.filter(lambda example: example['intent']==0)
        else:
            val_ood = val_dataset.filter(lambda example: example['intent'] not in label_ids)
            val_ood = val_ood.shuffle(seed=args.seed)
            val_ood = val_ood.sort('intent')
            val_ood = val_ood.shard(num_shards=num_shards*3, index=0)
            val_ood = val_ood.map(set_label_to_OOD)

            trainval_dataset = concatenate_datasets([val_ood, val_id, id])
            val_ood = concatenate_datasets([val_ood, val_id])

    else:
        trainval_dataset = concatenate_datasets([id, val_id])
        val_ood = val_id

    trainval_dataset.cast_column("intent", classlabel)

    #val_id.to_csv("val_id_csv")
    #train_dataset.to_csv("train_csv")
    #trainval_dataset.to_csv("trainval_csv")


    ########################################################### Test ###############################################################

    #Testdaten ID/OOD aufteilen
    test_dataset = datasets_dict['test']
    test_dataset = test_dataset.map(set_OOD_as_0)

    #Test OOD
    if ood_original is True:
        test_ood_dataset = test_dataset.filter(lambda example: example['intent'] == 0)
    else:
        test_ood_dataset = test_dataset.filter(lambda example: example['intent'] not in label_ids)
        test_ood_dataset = test_ood_dataset.map(set_label_to_OOD)
    test_ood_dataset = test_ood_dataset.cast_column("intent", classlabel)

    #Test ID
    test_id_dataset = test_dataset.filter(lambda example: example['intent'] in label_ids)
    test_id_dataset = test_id_dataset.map(set_label_to_ID)
    test_id_dataset = test_id_dataset.cast_column("intent", classlabel)

    #Test ganz
    test_dataset = concatenate_datasets([test_id_dataset, test_ood_dataset])

    



    #???HR PROBLEM: keien OOD Daten -> OOD Test trotzdem auf 0 -> man kann keine "normale" eval machen, sonder muss zwingend ood und id getrennt betrachten bzw über Thresholds arbeiten, da es die klasse ood nicht gibt

    train_dataset.to_csv('ichraste.csv')  

    #dev_dataset = train + dev_id
    return DatasetDict({'train': train_dataset, 'trainval':  trainval_dataset, 'val_id': val_id, 'val_ood': val_ood, 'test': test_dataset, 'test_ood': test_ood_dataset, 'test_id': test_id_dataset})


def load_clinc_with_ID_Augmentation(args):

    clinc_DatasetDict = load_clinc(args)
    #get Augmented Data
    train_data = load_dataset('csv', data_files={'train': ['/content/drive/MyDrive/Masterarbeit/ID_Augmentation/train_augmented.csv']})
    clinc_DatasetDict['train'] = train_data['train'].remove_columns("Unnamed: 0")
    print(clinc_DatasetDict)
    return clinc_DatasetDict

def load_clinc_with_Augmentation(args):
    
    clinc_DatasetDict = load_clinc(args)

    #für ohne OOD-Daten
    return clinc_DatasetDict

    def prepare_txt(example):

        #index und /t vor dem Satz entfernen
        example['text'] = re.sub(r'^.*?/t', '', example['text'])
        #. und ? als Satzzeichen entfernen
        example['text'] = example['text'].strip(".?")
        
        # label hinzufügen (sind alle OOD)
        example['intent'] = 0

        return example

    for datafile in ['/content/OOD-Framework/data/Augmentation/wiki.txt', '/content/OOD-Framework/data/Augmentation/subset_books.txt']:

        num_labels = get_num_labels(args)

        label_names, label_ids = get_labels(args)

        data_dict = load_dataset('text', data_files={'train': datafile})
        train_dataset = data_dict['train']
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset = train_dataset.shard(num_shards=70, index=0)
        train_dataset = train_dataset.map(prepare_txt)
        classlabel = ClassLabel(num_classes = num_labels, names = label_names)
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
