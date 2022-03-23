import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_imlm, finetune_std
from ood_detection import detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, get_num_labels, save_model

warnings.filterwarnings("ignore")

#ID: 1

# Todo:
# - Training args für IMLM

def main():

    #get args
    args = get_args()

    #save_path like Model/1/...
    if args.save_path:
        args.save_path = args.save_path + str(args.model_ID) + "/"

    #init WandB
    if args.wandb == "log":
        wandb.init(project=args.project_name, name=str(args.model_ID) + '-' + str(args.alpha) + "_" + args.loss)

    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    #Todo: set seeds?

    #get num_labels
    num_labels = get_num_labels(args)

    if args.task == "finetune":

        ##################### IMLM ###############################
        #Load Model
        print("Load model for IMLM...")
        model, config, tokenizer = set_model(args, num_labels)

        #Preprocess Data
        print("Preprocess Data for IMLM...")
        train_dataset, dev_dataset, datacollector = preprocess_data(args.dataset, args, num_labels, tokenizer, no_Dataloader=True, model_type='LanguageModeling')

        #Finetune IMLM + abspeichern
        print("Finetune IMLM...")
        trainer = finetune_imlm(args, model, train_dataset, dev_dataset, datacollector, tokenizer)
        args.model_name_or_path = args.save_path + "IMLM/"
        trainer.save_model(args.model_name_or_path)

        ##################### BCAD ###############################
        #Load Model for BCAD (args.model_name_or_path wurde geändert)
        print("Load model for BCAD...")
        model, config, tokenizer = set_model(args, num_labels)
        
        #Preprocess Data
        print("Preprocess Data for IMLM...")
        train_dataset, dev_dataset, test_id_dataset, test_ood_dataset = preprocess_data("clinc150_AUG", args, num_labels, tokenizer)
        
        #Finetune BCAD + abspeichern
        print("Finetune BCAD...")
        ft_model = finetune_std(args, model, train_dataset, dev_dataset)
        args.save_path = args.save_path + "IMLM_BCAD/"

        #save finetuned model
        #Model speichern
        if args.save_path:
            save_model(model, args.save_path)


    elif args.task == "ood_detection":

        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model, centroids-file und delta-file enthält!")
            return False

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args, num_labels)
        
        #Preprocess Data
        train_dataset, dev_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args.dataset, args, num_labels, tokenizer)

        
        #OOD-Detection
        print("Start OOD-Detection...")
        detect_ood(args, model, dev_dataset, test_id_dataset, test_ood_dataset)

if __name__ == "__main__":
    main()
