import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import re

datasets.logging.set_verbosity(datasets.logging.ERROR)


def preprocess_data(dataset_name, few_shot, num_labels, ood_data, tokenizer):

    print("Loading {}".format(dataset_name))
    if dataset_name == 'clinc150':
        raw_datasets = load_clinc(few_shot, num_labels, ood_data)
    if dataset_name == 'clinc150_AUG':
        raw_datasets = load_clinc_with_Augmentation(few_shot, num_labels, ood_data)
    else:
       raise NotImplementedError


    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #Columns anpassen
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("intent", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )
    test__ood_dataloader = DataLoader(
        tokenized_datasets["test_ood"], batch_size=8, collate_fn=data_collator
    )
    test_id_dataloader = DataLoader(
        tokenized_datasets["test_id"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader, test_id_dataloader, test__ood_dataloader


def load_clinc(few_shot, num_labels, ood_data):
    #few_shot: Anteil vom ursprünglichen Datensatz
    #num_labels: Anzahl label -> 2 oder 151
    #ood_data: 'zero' -> nur ID, sonst ID + OOD

    num_shards = int(100/few_shot)

    datasets_dict = load_dataset("clinc_oos", "small")

    def change_ood_label(example):
        #OOD = label 0
        if example['intent'] == 42:
            example['intent'] = 0
        elif example['intent'] > 42:
            example['intent'] = example['intent'] - 1
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

    #Validation Daten ID/OOD aufteilen
    val_dataset = datasets_dict['validation']
    val_dataset = val_dataset.map(change_ood_label)
    if ood_data == 'zero':
        val_dataset = val_dataset.filter(lambda example: example['intent'] != 0)
    
    #Testdaten ID/OOD aufteilen
    test_dataset = datasets_dict['test']
    test_dataset = test_dataset.map(change_ood_label)
    test_ood_dataset = test_dataset.filter(lambda example: example['intent'] == 0)
    test_id_dataset = test_dataset.filter(lambda example: example['intent'] != 0)

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

        data_dict = load_dataset('text', data_files={'train': datafile})
        train_dataset = data_dict['train']
        train_dataset = train_dataset.shuffle(seed=42)
        train_dataset = train_dataset.shard(num_shards=70, index=0)
        train_dataset = train_dataset.map(prepare_txt)
        train_dataset = train_dataset.rename_column("text", "intent")

        clinc_DatasetDict['train'] = concatenate_datasets([clinc_DatasetDict['train'], train_dataset])

    return clinc_DatasetDict
    
    