import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_std, finetune_ADB
from ood_detection import detect_ood, test_detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, get_num_labels, save_model, save_tensor
from accelerate import Accelerator

warnings.filterwarnings("ignore")

## ID 2

def main():

    #get args
    args = get_args()


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
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, dev_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)


if __name__ == "__main__":
    main()
