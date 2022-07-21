import torch
import wandb
import warnings
from utils.args import get_args
from data import load_clinc
from utils.utils import set_seed
from utils.utils_gold import create_gold_json
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


    #Preprocess Data
    print("Preprocess Data...")
    dataset_dict = load_clinc(args)
    dataset_dict['train'].to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/train.csv")
    dataset_dict['val_ood'].to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/validation.csv")
    dataset_dict['test'].to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/test.csv")

    create_gold_json(args)



       

if __name__ == "__main__":
    main()
