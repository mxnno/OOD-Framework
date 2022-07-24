import torch
import wandb
import warnings
import time
import copy



from model import  set_model
from finetune import finetune_std
from ood_detection_ood import detect_ood as detect_ood_with_ood
from utils.args import get_args
from utils.utils import set_seed, save_model
from accelerate import Accelerator
from model_temp import ModelWithTemperature
from ood_detection import detect_ood, detect_ood_DNNC
from data import preprocess_data, load_clinc
from utils.utils_DNNC import *
from evaluation import evaluate_method_combination

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

        #init WandB
        if args.wandb == "log":
            wandb.init(project=str(args.model_ID), name='{}_{}_{}_{}_{}'.format(args.id_data, args.ood_data, args.few_shot, int(args.num_train_epochs), args.seed))

        #Load Model
        print("Load model...")
        model, config, tokenizer = set_model(args)

        #Preprocess Data
        print("Preprocess Data...")
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset, eval_id, eval_ood = preprocess_data(args, tokenizer)
        

        #Pretrain SCL
        print("Pretrain SCL (margin/similarity) ...")
        ft_model, best_epoch =  finetune_std(args, model, train_dataset, eval_id, eval_ood, accelerator)

        #Finetune auf CE oder LMCL + abspeichern
        print("Finetune CE/LMCL...")
        #model.config.loss = ''
        #ft_model =  finetune_std(args, model, train_dataset, dev_id_dataset, accelerator)
        if args.save_path != "debug":
            save_model(ft_model, args, best_epoch)


    elif args.task == "ood_detection":
        
        if not args.save_path:
            print("Bitte einen Pfad angeben, der ein Model enthält!")
            return False


        #Load Model
        print("Load model...")
        sPath = args.model_name_or_path
        model, config, tokenizer = set_model(args, sPath)

        #Preprocess Data
        #dev_dataset = train + dev_id
        train_dataset, traindev_dataset, dev_id_dataset, dev_ood_dataset, test_dataset, test_id_dataset, test_ood_dataset, eval_id, eval_ood = preprocess_data(args, tokenizer)
        

        #Temp für Softmax ermitteln
        # Now we're going to wrap the model with a decorator that adds temperature scaling
        temp_model = ModelWithTemperature(model)

        # Tune the model temperature, and save the results
        best_temp = temp_model.set_temperature(dev_id_dataset)

        #OOD-Detection
        print("Start OOD-Detection...")
        if args.ood_data != "zero":
            detect_ood_with_ood(args, model, train_dataset, traindev_dataset, dev_ood_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp)
        else:
            name_2, in2, out2 = detect_ood(args, model, train_dataset, traindev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp)

        #für 2m,3:
        # - 10 nromale
        # - 5 mit k3
        # - 5 mit k2
        # - 5 mit kalle
        sPath = "/content/drive/MyDrive/Masterarbeit/Trainierte_Modelle/3/travel_zero_5_6_222"
        model, config, tokenizer = set_model(args, path=sPath)
        temp_model = ModelWithTemperature(model)
        best_temp = temp_model.set_temperature(dev_id_dataset)
        name3, in2, out2 = detect_ood(args, model, train_dataset, traindev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp)

        #NLI
        args.model_ID = 8
        sPath ="/content/drive/MyDrive/Masterarbeit/Trainierte_Modelle/8/travel_zero_5_2_222"
        model, config, tokenizer = set_model(args,path=sPath)
        dataset_dict  = load_clinc(args)
        train_dataset = dataset_dict['train']
        train_dataset.to_csv("train_dataset_8.csv")
        test_id = dataset_dict['test_id']
        test_ood = dataset_dict['test_ood']
        test_id.to_csv("test_id_dataset_8.csv")
        test_ood.to_csv("test_od_dataset_8.csv")
        time.sleep(2)
        test_data_id, test_data_ood = load_intent_datasets("test_id_dataset_8.csv", "test_od_dataset_8.csv", True)
        train_data, _ = load_intent_datasets("train_dataset_8.csv", "train_dataset_8.csv", True)
        
        #OOD-Detection
        print("Start OOD-Detection...")
        name8, in8, out8 = detect_ood_DNNC(args, model, tokenizer, train_data, test_data_id, test_data_ood)


        #ADB
        args.model_ID = 14
        sPath ="/content/drive/MyDrive/Masterarbeit/Trainierte_Modelle/14/travel_zero_5_30_222"
        model, config, tokenizer = set_model(args, path=sPath)
        #centroids holen
        centroids = torch.load(args.model_name_or_path + "/centroids.pt")
        #delta holen
        delta = torch.load(args.model_name_or_path + "/delta.pt")
        name14, in14, out14 = detect_ood(args, model, train_dataset, traindev_dataset, dev_id_dataset, test_id_dataset, test_ood_dataset, best_temp=best_temp, centroids=centroids, delta=delta)


        combis1 = ["2", "3", "8", "14"]
        combis2 = ["2", "3", "8", "14"]
        combis3 = ["2", "3", "8", "14"]

        n_list = []
        i_list = []
        o_list = []

        for x in combis1:
            combis2.pop(0)
            combis3.pop(0)

            cHelp = copy.deepcopy(combis3)
            for y in combis2:
                cHelp.pop(0)
                for z in cHelp:

                    nx = globals()["name" + x]
                    ix = globals()["in" + x]
                    ox = globals()["out" + x]

                    ny = globals()["name" + y]
                    iy = globals()["in" + y]
                    oy = globals()["out" + y]

                    nz = globals()["name" + z]
                    iz = globals()["in" + z]
                    oz = globals()["out" + z]

                    #a ist pred_in mit 450 werten
                    for i, a in enumerate(ix):
                        for j, b in enumerate(iy):
                            for k, c in enumerate(iz):

                                d = a + b + c
                                d = np.where(d>=2, 1, 0)
                                i_list.append(d)
                                n_list.append(nx[i] + "|" + ny[j] + "|" + nz[k])
                    
                    for i, a in enumerate(ox):
                        for j, b in enumerate(oy):
                            for k, c in enumerate(oz):

                                d = a + b + c
                                d = np.where(d>=2, 1, 0)
                                o_list.append(d)

        evaluate_method_combination(args, n_list, i_list, o_list, "best")




                        




if __name__ == "__main__":
    main()
