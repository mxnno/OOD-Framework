import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_std, finetune_ADB
from ood_detection import detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, get_num_labels, save_model, save_tensor, get_save_path

warnings.filterwarnings("ignore")

## ID 14
#Todo
# - Nehmen aus Trainingsdaten manche Klassen raus (Open Intent Detection) -> bei mir auch???
# -> ADB Evaluation nochmal anschauen ob das so passt

def main():

    #get args
    args = get_args()

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

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args, num_labels)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, dev_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args.dataset, args, num_labels, tokenizer)

        #Finetune Std + abspeichern
        # learn intent representation mit softmax loss
        print("Finetune...")
        ft_model =  finetune_std(args, model, train_dataset, dev_dataset)
        if args.save_path != "debug":
            save_model(ft_model, args)

        #Finetune ADB + abspeichern
        print("Finetune ADB...")
        ft_model, centroids, delta = finetune_ADB(args, ft_model, train_dataset, dev_dataset)
        if args.save_path != "debug":
            args.save_path = get_save_path(args).replace("/14/", "/14/ADB")
            save_model(ft_model, args)
            save_tensor(centroids, args.save_path)
            save_tensor(delta, args.save_path)
        else:
            detect_ood(args, ft_model, dev_dataset, test_id_dataset, test_ood_dataset, centroids=centroids, delta=delta)


    elif args.task == "ood_detection":
        
        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model, centroids-file und delta-file enthält!")
            return False

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args, num_labels)

        #centroids holen
        centroids = torch.load(args.model_name_or_path + "/centroids.pt")
        #delta holen
        delta = torch.load(args.model_name_or_path + "/delta.pt")

        #Preprocess Data
        train_dataset, dev_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args.dataset, args, num_labels, tokenizer)

        #OOD-Detection
        print("Start OOD-Detection...")
        detect_ood(args, model, dev_dataset, test_id_dataset, test_ood_dataset, centroids=centroids, delta=delta)

if __name__ == "__main__":
    main()
