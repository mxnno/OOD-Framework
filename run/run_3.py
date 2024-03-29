import torch
import wandb
import warnings

from model import  set_model
from finetune import finetune_std
from ood_detection import detect_ood
from ood_detection_ood import detect_ood as detect_ood_with_ood
from utils.args import get_args
from data import preprocess_data
from utils.utils import set_seed, save_model
from accelerate import Accelerator
from model_temp import ModelWithTemperature


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
            wandb.init(project=str(args.model_ID), name='{}_{}_{}_{}_{}'.format(args.id_data, args.ood_data, args.few_shot, int(args.num_train_epochs), args.seed))

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset, eval_id, eval_ood = preprocess_data(args, tokenizer)


        #### Ablauf:
        # 1. similarity-contrastive-augm mit ce -> 20
        # 2. normales loss mit ce -> 1
        # 3. normales loss mit lmcl -> 19 

        # Die Anzahl der Epochen noch testen!

        #Pretrain SCL
        print("Pretrain SCL ...")
        model.config.loss = 'similarity-contrastive-augm'
        model.config.loss_std = "lmcl_1"
        ft_model, best_epoch =  finetune_std(args, model, train_dataset, eval_id, eval_ood, accelerator, num_epochs_x = 20)

        
        #Finetune auf CE oder LMCL + abspeichern
        print("Finetune CE/LMCL...")
        #eine epoche mit lmcl, dann ce
        model.config.loss = ''
        model.config.loss_std = "lmcl_1"
        ft_model, best_epoch =  finetune_std(args, model, train_dataset, eval_id, eval_ood, accelerator, num_epochs_x = 1)
        model.config.loss_std = "lmcl_2"
        ft_model, best_epoch =  finetune_std(args, model, train_dataset, eval_id, eval_ood, accelerator, num_epochs_x = 20)
        if args.save_path != "debug":
            save_model(ft_model, args, best_epoch)

    elif args.task == "ood_detection":
        
        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model enthält!")
            return False


        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        #dev_dataset = train + dev_id
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset, eval_id, eval_ood = preprocess_data(args, tokenizer)
        

        #Temp für Softmax ermitteln
        # Now we're going to wrap the model with a decorator that adds temperature scaling
        temp_model = ModelWithTemperature(model)

        # Tune the model temperature, and save the results
        best_temp = temp_model.set_temperature(traindev_dataset)


        #OOD-Detection
        print("Start OOD-Detection...")
        if args.ood_data != "zero":
            detect_ood_with_ood(args, model, train_dataset, traindev_dataset, dev_ood_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp)
        else:
            detect_ood(args, model, train_dataset, traindev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp)

if __name__ == "__main__":
    main()
