import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_std, finetune_ADB
from ood_detection import detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, get_num_labels, save_model, save_tensor
from accelerate import Accelerator

warnings.filterwarnings("ignore")

## ID 2

def main():

    #get args
    args = get_args()

    #save_path like Model/1/...
    if args.save_path:
        args.save_path = args.save_path + str(args.model_ID) + "/"

    #init WandB
    if args.wandb == "log":
        wandb.init(project=args.project_name, name=str(args.model_ID) + '-' + str(args.alpha) + "_" + args.loss)

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

    #get num_labels
    num_labels = get_num_labels(args)

    if args.task == "finetune":

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args, num_labels)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, dev_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args.dataset, args, num_labels, tokenizer)

        #Finetune Std + abspeichern
        print("Finetune...")
        ft_model =  finetune_std(args, model, train_dataset, dev_dataset, accelerator)
        if args.save_path != "debug":
            save_model(ft_model, args)

    elif args.task == "ood_detection":
        
        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model enthält!")
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
