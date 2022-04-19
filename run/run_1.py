import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_imlm, finetune_std
from ood_detection import detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, get_num_labels, save_model, get_save_path
from accelerate import Accelerator

warnings.filterwarnings("ignore")

#ID: 1

# Todo:
# - Training args für IMLM

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


    if args.task == "finetune":

        #init WandB
        if args.wandb == "log":
            wandb.init(project=args.project_name, name=str(args.model_ID) + '-' + str(args.alpha) + "_" + args.loss)

        ##################### IMLM ###############################
        #Load Model
        print("Load model for IMLM...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data for IMLM...")
        train_dataset, dev_dataset, datacollector = preprocess_data(args, tokenizer, no_Dataloader=True, model_type='LanguageModeling')

        #Finetune IMLM + abspeichern
        print("Finetune IMLM...")
        trainer = finetune_imlm(args, model, train_dataset, dev_dataset, datacollector, tokenizer)
        args.save_path = get_save_path(args).replace("/1/", "/1/IMLM/")
        args.model_name_or_path = args.save_path
        trainer.save_model(args.save_path)

        ##################### BCAD ###############################
        #Load Model for BCAD (args.model_name_or_path wurde geändert)
        print("Load model for BCAD...")
        model, config, tokenizer = set_model(args)
        
        #Preprocess Data
        print("Preprocess Data for IMLM...")
        args.dataset = "clinc150_AUG"
        train_dataset, dev_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)
        
        #Finetune BCAD + abspeichern
        print("Finetune BCAD...")
        ft_model = finetune_std(args, model, train_dataset, dev_dataset, accelerator)
        args.save_path = get_save_path(args).replace("/1/", "/1/IMLM_BCAD/")

        #save finetuned model
        #Model speichern
        if args.save_path != "debug":
            save_model(ft_model, args)



    elif args.task == "ood_detection":

        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model, centroids-file und delta-file enthält!")
            return False

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)
        
        #Preprocess Data
        train_dataset, dev_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)

        
        #OOD-Detection
        print("Start OOD-Detection...")
        detect_ood(args, model, dev_dataset, test_id_dataset, test_ood_dataset)

if __name__ == "__main__":
    main()
