import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_std
from ood_detection import detect_ood, test_detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, save_model
from accelerate import Accelerator

warnings.filterwarnings("ignore")

## ID 2

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

        #Erst Pretrain auf SCL dann Finetune auf CE oder LMCL

        #init WandB
        if args.wandb == "log":
            wandb.init(project=args.project_name, name=str(args.model_ID) + '-' + str(args.alpha) + "_" + args.loss)

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, dev_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)

        #Pretrain SCL
        print("Pretrain SCL ...")
        args.num_train_epoch = 100
        model.config.loss = 'similarity-contrastive-augm'
        ft_model =  finetune_std(args, model, train_dataset, dev_dataset, accelerator)

        args.num_train_epoch = 3
        #Finetune auf CE oder LMCL + abspeichern
        print("Finetune CE/LMCL...")
        model.config.loss = ''
        ft_model =  finetune_std(args, model, train_dataset, dev_dataset, accelerator)
        if args.save_path != "debug":
            save_model(ft_model, args)

    elif args.task == "ood_detection":
        
        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model enthält!")
            return False

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        train_dataset, dev_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)

        label_list = []
        for batch in train_dataset:
            label_list += list(batch["labels"].cpu().detach().numpy())

        print(label_list)
        #OOD-Detection
        #print("Start OOD-Detection...")
        detect_ood(args, model, train_dataset, test_id_dataset, test_ood_dataset)
        #test_detect_ood(args, model, dev_dataset, test_dataset)

if __name__ == "__main__":
    main()
