import torch
import wandb
import warnings
import time
import os

from model import  set_model
from finetune import finetune_std, finetune_ADB, finetune_DNNC
from ood_detection import detect_ood, detect_ood_DNNC
from utils.args import get_args
from data import preprocess_data, load_clinc
from utils.utils import set_seed, get_num_labels, save_model
from utils.utils_DNNC import *
from accelerate import Accelerator

warnings.filterwarnings("ignore")

## ID 14
#Todo
# - Nehmen aus Trainingsdaten manche Klassen raus (Open Intent Detection) -> bei mir auch???
# -> ADB Evaluation nochmal anschauen ob das so passt

def main():

    #get args
    args = get_args()

    #Accelerator
    if args.tpu == "tpu":
        accelerator = Accelerator()
        args.device = accelerator.device
    else:
        #set device
        accelerator = None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
        set_seed(args)
        #Todo: set seeds?

    do_lower_case = True


    if args.task == "finetune":

        #init WandB
        if args.wandb == "log":
            wandb.init(project=str(args.model_ID), name='{}_{}_{}_{}_{}'.format(args.id_data, args.ood_data, args.few_shot, int(args.num_train_epochs), args.seed))

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        # #Preprocess NLI Data -> load 
        # print("Preprocess NLI Data...")
        #nli_train_examples = load_nli_examples('/content/drive/MyDrive/Masterarbeit/DNNC/nli/all_nli.train.txt', True)
        # nli_dev_examples = load_nli_examples('/content/drive/MyDrive/Masterarbeit/DNNC/nli/all_nli.dev.txt', True)

        #NLI Pretrain + abspeichern


        #get raw Dataset
        dataset_dict  = load_clinc(args)
        train_dataset = dataset_dict['train']
        train_dataset.to_csv("train_dataset.csv")
        

        #load_intent_datasets -> list with examples e
        train_data, _ = load_intent_datasets("train_dataset.csv", "train_dataset.csv", do_lower_case)
        
        #Eval
        eval_id_dataset = dataset_dict['val_test_id']
        eval_ood_dataset = dataset_dict['val_test_ood']
        eval_id_dataset.to_csv("eval_id.csv")
        eval_ood_dataset.to_csv("eval_ood.csv")
        eval_id, eval_ood = load_intent_datasets("eval_id.csv", "eval_ood.csv", do_lower_case)

        #Test
        test_id = dataset_dict['test_id']
        test_ood = dataset_dict['test_ood']
        test_id.to_csv("test_id_dataset.csv")
        test_ood.to_csv("test_od_dataset.csv")
        test_data_id, test_data_ood = load_intent_datasets("test_id_dataset.csv", "test_od_dataset.csv", do_lower_case)

        time.sleep(2)

        # NLI Examples erstellen
        nli_train, nli_dev = create_nli_examples(args, train_data)

        # Tokenization passiert in finetune Methode
        #Finetune:
        print("Finetune DNNC...")
        ft_model, best_epoch = finetune_DNNC(args, model, tokenizer, nli_train, train_data, eval_id, eval_ood)

        if args.save_path != "debug":
            save_model(ft_model, args, best_epoch)

    elif args.task == "ood_detection":
        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        dataset_dict  = load_clinc(args)
        train_dataset = dataset_dict['train']
        train_dataset.to_csv("train_dataset_8.csv")
        test_id = dataset_dict['test_id']
        test_ood = dataset_dict['test_ood']
        test_id.to_csv("test_id_dataset_8.csv")
        test_ood.to_csv("test_od_dataset_8.csv")
        time.sleep(2)
        test_data_id, test_data_ood = load_intent_datasets("test_id_dataset_8.csv", "test_od_dataset_8.csv", do_lower_case)
        train_data, _ = load_intent_datasets("train_dataset_8.csv", "train_dataset_8.csv", do_lower_case)
        
        #OOD-Detection
        print("Start OOD-Detection...")
        detect_ood_DNNC(args, model, tokenizer, train_data, test_data_id, test_data_ood)



if __name__ == "__main__":
    main()
