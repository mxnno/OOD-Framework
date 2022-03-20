import argparse
import torch

import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from model import RobertaForSequenceClassification
from finetune import finetune, finetune_ADB
from ood_detection import detect_ood
from data import preprocess_data
import wandb
import warnings
from utils.utils import set_seed

warnings.filterwarnings("ignore")


task_to_labels = {
    'full': 151,
    'unlabeled': 2,
    'zero': 2
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['finetune','ood_detection'])
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--model_ID", type=int)
    parser.add_argument("--dataset", type=str, default="clinc150")
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--ood_data", default="full", type=str, choices=['full','zero'])
    parser.add_argument("--id_data", default="full", type=str, choices=['full', 'unlabeled','zero'])
    parser.add_argument("--few_shot", default="100", type=int)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--save_path", type=str)

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, choices=['margin-contrastive', 'similarity-contrastive', 'default'], default='default')
    args = parser.parse_args()

    wandb.init(project=args.project_name, name=str(args.model_ID) + '-' + str(args.alpha) + "_" + args.loss)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    #Todo: set seeds?
    print("#############")
    print("Load model...")
    num_labels = task_to_labels[args.id_data]
    if args.model_name_or_path == ('roberta-base'):
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        model.to(device)
    else:
        # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        # config.gradient_checkpointing = True
        # config.alpha = args.alpha
        # config.loss = args.loss
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)
        # model.to(device)
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        model.to(device)

    print("##################")
    print("Preprocess Data...")
    train_dataset, dev_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args.dataset, args.few_shot, num_labels, args.ood_data, tokenizer)

    if args.task == "finetune":
        print("###################")
        print("Start finetuning...")
        ft_model = finetune(args, model, train_dataset, dev_dataset)
        
        if args.model_ID == 1:
            ft_model.to(device)
            print("#######################")
            print("Start finetuning ADB...")
            ft_model, centroids, delta = finetune_ADB(args, ft_model, train_dataset, dev_dataset)
            
            print("######################")
            print("Start OOD-Detection...")
            detect_ood(args, model, dev_dataset, test_id_dataset, test_ood_dataset, centroids=centroids, delta=delta)

        if args.save_path:
            ft_model.save_pretrained(args.save_path)
            print("Model saved at: " + args.save_path)

    elif args.task == "ood_detection":
        print("######################")
        print("Start OOD-Detection...")
        detect_ood(args, model, dev_dataset, test_id_dataset, test_ood_dataset)




if __name__ == "__main__":
    main()
