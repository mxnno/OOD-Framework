import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_imlm, finetune_std
from ood_detection_ood import detect_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, get_num_labels, save_model, get_save_path
from accelerate import Accelerator
from model_temp import ModelWithTemperature


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
            wandb.init(project=str(args.model_ID), name='{}_{}_{}_{}_{}'.format(args.id_data, args.ood_data, args.few_shot, int(args.num_train_epochs), args.seed))

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data for IMLM...")

        ##PROBLEM:
        ##IMLM mit zero OOD -> nur 1 Klasse
        # entweder zweite alibi klasse machen oder 14 Klassen??

        id_data_backup = args.id_data


        args.ood_data = 'zero'
        args.id_data = 'travel'
        train_dataset, dev_id_dataset, datacollector = preprocess_data(args, tokenizer, no_Dataloader=True, model_type='LanguageModeling')

        #Finetune IMLM + abspeichern
        print("Finetune IMLM...")
        trainer = finetune_imlm(args, model, train_dataset, dev_id_dataset, datacollector, tokenizer)
        save_path = get_save_path(args, 0).replace("/1/", "/1/IMLM/")
        #args.model_name_or_path = save_path
        trainer.save_model(save_path)

        print(save_path)
        ##### WICHTIG: Verhältniss Epochen BCAD=2 zu IMLM=10 

        ##################### BCAD ###############################
        #Load Model for BCAD (args.model_name_or_path wurde geändert)
        print("Load model for BCAD...")
        args.ood_data = "augm"
        args.id_data = id_data_backup
        model, config, tokenizer = set_model(args, path=save_path)
        
        #Preprocess Data
        print("Preprocess Data for BCAD...")
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset, eval_id, eval_ood = preprocess_data(args, tokenizer, special_dataset="clinc150_AUG")

        #Finetune BCAD + abspeichern
        print("Finetune BCAD...")
        ft_model, best_epoch = finetune_std(args, model, train_dataset, eval_id, eval_ood, accelerator)

        #save finetuned model
        #Model speichern
        if args.save_path != "debug":
            save_model(ft_model, args, best_epoch)



    elif args.task == "ood_detection":

        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model enthält!")
            return False

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args, path = args.model_name_or_path)

        #Preprocess Data
        #dev_dataset = train + dev_id

        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset, eval_id, eval_ood = preprocess_data(args, tokenizer)
        

        #Temp für Softmax ermitteln
        # Now we're going to wrap the model with a decorator that adds temperature scaling
        temp_model = ModelWithTemperature(model)

        # Tune the model temperature, and save the results
        #best_temp = temp_model.set_temperature(dev_id_dataset)
        best_temp = 1.5

        #OOD-Detection
        print("Start OOD-Detection...")
        detect_ood(args, model, train_dataset, traindev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp)
if __name__ == "__main__":
    main()
