import datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

datasets.logging.set_verbosity(datasets.logging.ERROR)


def load_dataset(dataset_name, few_shot, num_labels, ood_data, tokenizer):

    print("Loading {}".format(dataset_name))
    if dataset_name == 'clinc150':
        raw_datasets = load_clinc(few_shot, num_labels, ood_data)
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

    #Trainingsdaten ID/OOD aufteilen + verkleinern
    dataset = datasets_dict['train']
    ood = dataset.filter(lambda example: example['intent'] == 42)
    id = dataset.filter(lambda example: example['intent'] != 42)

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
    if ood_data == 'zero':
        val_dataset = val_dataset.filter(lambda example: example['intent'] != 42)
    
    #Testdaten ID/OOD aufteilen
    test_dataset = datasets_dict['test']
    test_ood_dataset = test_dataset.filter(lambda example: example['intent'] == 42)
    test_id_dataset = test_dataset.filter(lambda example: example['intent'] != 42)

    #Falls 2 Klassen 
    def change_label(example):
        if example['intent'] == 42:
            example['intent'] = 0
        else:
            example['intent'] = 1
            return example

    if num_labels == 2:
        train_dataset = train_dataset.map(change_label)
        val_dataset = val_dataset.map(change_label)
        test_ood_dataset = test_ood_dataset.map(change_label)
        test_id_dataset = test_id_dataset.map(change_label)

    return DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test_ood': test_ood_dataset, 'test_id': test_id_dataset})

