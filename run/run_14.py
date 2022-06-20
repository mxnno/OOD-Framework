import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_std, finetune_ADB
from ood_detection import detect_ood
from utils.args import get_args
from data import preprocess_data
from accelerate import Accelerator
from utils.utils import set_seed, get_num_labels, save_model, save_tensor, get_save_path
from model_temp import ModelWithTemperature


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

    if args.task == "finetune":

        #init WandB
        if args.wandb == "log":
            wandb.init(project=str(args.model_ID), name='{}_{}_{}_{}_{}'.format(args.id_data, args.ood_data, args.few_shot, int(args.num_train_epochs), args.seed))


        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)

        #Finetune Std + abspeichern
        # learn intent representation mit softmax loss
        print("Finetune...")
        ft_model =  finetune_std(args, model, train_dataset, dev_id_dataset, accelerator)
        if args.save_path != "debug":
            save_model(ft_model, args)

        #Finetune ADB + abspeichern
        print("Finetune ADB...")
        ft_model, centroids, delta = finetune_ADB(args, ft_model, train_dataset, dev_id_dataset)
        if args.save_path != "debug":
            args.save_path = get_save_path(args)
            save_model(ft_model, args)
            save_tensor(args, centroids, "/centroids.pt")
            save_tensor(args, delta, "/delta.pt")
        else:
            detect_ood(args, ft_model, dev_id_dataset, test_id_dataset, test_ood_dataset, centroids=centroids, delta=delta)


    elif args.task == "ood_detection":
        
        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model, centroids-file und delta-file enthält!")
            return False

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #centroids holen
        centroids = torch.load(args.model_name_or_path + "/centroids.pt")
        #delta holen
        delta = torch.load(args.model_name_or_path + "/delta.pt")


        #Preprocess Data
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset = preprocess_data(args, tokenizer)

        #Temp für Softmax ermitteln
        # Now we're going to wrap the model with a decorator that adds temperature scaling
        temp_model = ModelWithTemperature(model)

        # Tune the model temperature, and save the results
        best_temp = temp_model.set_temperature(dev_id_dataset)

        #OOD-Detection
        print("Start OOD-Detection...")
        detect_ood(args, model, dev_id_dataset, test_id_dataset, test_ood_dataset, centroids=centroids, delta=delta)

if __name__ == "__main__":
    main()
