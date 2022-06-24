from utils.args import get_args
import os

#get args
args = get_args()


#FÜR STEP 0
id_list = [0]
path = '/content/drive/MyDrive/Masterarbeit/Results/' + args.model_ID + '/'

#FÜR STEP 1

#1: IMLM bzw IMLM_BCAD + STD Files
#2: STD Files
#3: STD Files
#8: nur 1 File
#14: nur 1 File

id_list = [1, 2, 3, 8, 14]

for id in id_list:

    if id == 1:
        extra_Ordner = "IMLM/"
    else:
        extra_Ordner = ""

    path = '/content/drive/MyDrive/Masterarbeit/Results/' + id + '/' + extra_Ordner

    for folder in os.listdir(path):
        if folder.startswith("travel_zero") or folder.startswith("banking_zero"):
            path = os.path.join(path, folder)
            few_shot = folder.split("_")[2]
            epochs = folder.split("_")[3]
            seed = folder.split("_")[4]



            #ID 8 und und 14 nur 1 File
            if id == 8:
                filename = 'results_NLI.csv'
            if id == 14:
                filename = 'results_ADB.csv'



            
        else:
            continue

