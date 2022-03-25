import torch
import warnings

from model import  set_model
from utils.args import get_args
from utils.utils import set_seed, get_num_labels
from accelerate import Accelerator

import torch.nn.functional as F
import torch 
import numpy as np


warnings.filterwarnings("ignore")

## ID 2

def main():

    #get args
    args = get_args()

    #Accelerator
    if args.accelerator is True:
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

    num_labels = get_num_labels(args)
    #load model
    print("Load model...")
    model, config, tokenizer = set_model(args, num_labels)

    
    input_string = "" 
    while input_string != "exit":

        input_string = input("'exit' zum abbrechen:")
        tokenized = tokenizer(input_string)


        tokenized['input_ids'] = torch.tensor(tokenized['input_ids'], device=args.device).unsqueeze(0)
        tokenized['attention_mask'] = torch.tensor(tokenized['attention_mask'], device=args.device).unsqueeze(0)

        outputs = model(**tokenized)
        logits = outputs[0]
        softmax_score, softmax_label = F.softmax(logits, dim=-1).max(-1)
        print("Score: " + str(softmax_score.detach().cpu().numpy()))
        print("Label: " + str(softmax_label.detach().cpu().numpy()))
        logits = logits.detach().cpu().numpy()
        
        print(np.argmax(logits, axis=1)[0])





    

if __name__ == "__main__":
    main()
