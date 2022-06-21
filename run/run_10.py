import torch
import warnings
from utils.args import get_args
from data import load_clinc
from utils.utils import set_seed, get_labels
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
    _ , label_ids = get_labels(args)
    dataset_dict = load_clinc(args)
    
    #Train
    train = dataset_dict['train'].rename_column("intent", "label")
    train = train.rename_column("text", "data")
    train.to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/kFolden/clinc/train/train.csv")
    
    #Dev
    #split in id und ood
    id_dev = dataset_dict['val_id'].rename_column("intent", "label")
    id_dev = id_dev.rename_column("text", "data")
    id_dev.to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/kFolden/clinc/dev/id_dev.csv")

    #21.06 geändert, da val_ood für early stopping genutzt wird
    #dev soll keine ood Daten haben -> val_ood = val_id (in data.py) nochmal prüfen
    ood_dev = dataset_dict['val_ood'].rename_column("intent", "label")
    ood_dev = ood_dev.rename_column("text", "data")
    ood_dev.to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/kFolden/clinc/dev/ood_dev.csv")

    #Test
    id_test = dataset_dict['test_id'].rename_column("intent", "label")
    id_test = id_test.rename_column("text", "data")

    id_test.to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/kFolden/clinc/test/id_test.csv")

    ood_test = dataset_dict['test_ood'].rename_column("intent", "label")
    ood_test = ood_test.rename_column("text", "data")
    ood_test.to_csv("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/kFolden/clinc/test/ood_test.csv")


     
if __name__ == "__main__":
    main()
