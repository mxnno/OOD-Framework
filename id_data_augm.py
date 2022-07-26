#source https://github.com/ml6team/quick-tips/blob/main/nlp/2021_11_25_augmentation_lm/nlp_augmentation_lm.ipynb
#teuer -> 0,66 cent pro Anfrage

import random
from datasets import load_from_disk, Dataset, ClassLabel, concatenate_datasets
import json
import random
import requests
from data import preprocess_data
from model import  set_model
from utils.args import get_args
from utils.utils import get_labels
import torch
from utils.utils import set_seed, save_model



def id_data_augm():


    n = 20
    ood_augm = True

    
    #get args
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    #Load Model
    print("Load model...")
    _ , _ , tokenizer = set_model(args)

    #Preprocess Data
    print("Preprocess Data..._")
    train_dataset_full = preprocess_data(args, tokenizer,id_augm=True)

    
    label_names, label_ids = get_labels(args)
    label_exludes = get_label_excludes(label_names)
    print(label_exludes)

    if ood_augm is True:


        train_dataset = train_dataset_full
        n = 75

        # define the number of synthetic samples to generate
        new_texts = []
        new_labels = []
        api_key = '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'
        #mail: yelax88626@altpano.com, pw 1q2w3e4r
        headers = {'Authorization' : 'Bearer ' + api_key,
                    'Content-type':'application/json', 
                    'Accept':'application/json'}

        print("start OOD AUGM generation")
        iter = 0
        while iter < n:
            
            # select two random samples from training set
            #texts, labels = get_15_random_samples(args, train_dataset)
            text1, label1, text2, label2, = get_two_random_samples(train_dataset)
            # create the prompt
            #prompt = get_prompt_ood_2(texts, labels, label_names)
            prompt = get_prompt_ood(text1, label1, text2, label2, label_names)
            # send a post request to gpt-3 using the prompt
            response = requests.post('https://api.openai.com/v1/engines/davinci/completions',
                                        headers=headers,
                                        data = json.dumps({"prompt": prompt, 
                                                        "max_tokens": 30,
                                                        "temperature": 0.9,
                                                        "top_p": 0.95}))

            # get response and extract the generated text and label
            # the generated output will be in the form "<text> (Sentiment: <label>)"

            data = response.json()['choices'][0]['text'].split('\n')[0].split('(Domain:')

            if len(data) < 2:
                # the format of the response is invalid
                continue

            text = data[0]
            label = data[1].split(')')[0].strip()
            
            if label in label_exludes or 'travel' in label:
                pass
            else:
                print(label)
                print(text)
                new_texts.append(text)
                new_labels.append(0)
                iter += 1
            

            

        # define the synthetic dataset and save it to disk so as to prevent sending 
        # many api requests
        ood_dataset = Dataset.from_dict({'text': new_texts, 'intent': new_labels})
        ood_dataset.to_csv("/content/drive/MyDrive/Masterarbeit/ID_Augmentation/gpt3/ood_augm.csv")
        

    else:    

        #ID AUGMENTATION

        for label_id,  label_name in enumerate(label_names):

            print("Label: " + str(label_name))

            train_dataset = train_dataset_full.filter(lambda e: e['intent'] == label_id)

            # define the number of synthetic samples to generate
            new_texts = []
            new_labels = []
            api_key = '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'
            headers = {'Authorization' : 'Bearer ' + api_key,
                        'Content-type':'application/json', 
                        'Accept':'application/json'}

            print("start ID AUGM generation")
            iter = 0
            while iter < n:
                # select two random samples from training set
                text1, label1, text2, label2 = get_two_random_samples(train_dataset)
                # create the prompt
                prompt = get_prompt(text1, label1, text2, label2, label_names)
                # send a post request to gpt-3 using the prompt
                response = requests.post('https://api.openai.com/v1/engines/davinci/completions',
                                            headers=headers,
                                            data = json.dumps({"prompt": prompt, 
                                                            "max_tokens": 30,
                                                            "temperature": 0.9,
                                                            "top_p": 0.95}))

                # get response and extract the generated text and label
                # the generated output will be in the form "<text> (Sentiment: <label>)"
                print(response)
                print(response.json())
                data = response.json()['choices'][0]['text'].split('\n')[0].split('(Domain:')

                if len(data) < 2:
                    # the format of the response is invalid
                    continue

                text = data[0]
                label = data[1].split(')')[0].strip()


                if label != label_name:
                # the format of the response is invalid
                    continue
                

                new_texts.append(text)
                if ood_augm:
                    new_labels.append(0)
                else:
                    new_labels.append(label_names.index(label))
                    
                iter += 1

            # define the synthetic dataset and save it to disk so as to prevent sending 
            # many api requests
            synthetic_ds = Dataset.from_dict({'text': new_texts, 'intent': new_labels})
            synthetic_ds.save_to_disk('./data/id_augm/gpt-3/' + str(label_id))

            if label_id == 0:
                full_aug_ds = synthetic_ds
            else:
                full_aug_ds = concatenate_datasets([full_aug_ds, synthetic_ds])

        full_aug_ds.save_to_disk('./data/id_augm/gpt-3/full_id_augm/' )
        #local
        full_aug_ds.to_csv('id_augm_' + str(args.seed) + '.csv')
        #drive
        drive_path = "/content/drive/MyDrive/Masterarbeit/ID_Augmentation/gpt3/ID/"
        full_aug_ds.to_csv(drive_path + str(args.id_data) + "_" + str(args.few_shot) + "to" + str(n) + "_" + str(args.seed) + ".csv")





def get_two_random_samples(train_dataset):
    # define a function that returns two random samples from the train set.
    s1, s2 = random.sample(range(0, len(train_dataset)), 2)
    return train_dataset['text'][s1], train_dataset['intent'][s1], train_dataset['text'][s2], train_dataset['intent'][s2]

def get_15_random_samples(args, train_dataset):
    # define a function that returns two random samples from the train set.
    
    texts = []
    labels = []
    for idx in range(0, len(train_dataset), args.few_shot):
        rand = random.randint(0, args.few_shot-1)
        texts.append(train_dataset['text'][idx + rand])
        labels.append(train_dataset['intent'][idx + rand])

    return texts, labels

def get_prompt(text1, label1, text2, label2, label_names):
    # define a function that takes as input two samples and generates the prompt
    # that we should pass to the GPT-3 language model for completion.
    description = "Each item in the following list contains a chatbot question and its Sub-Domain. Sub-Domain is " + str(label_names[label1]) + "."
    #description = "Each item in the following list contains a chatbot question and its Sub-Domain. Sub-Domain is one of 'exchange_rate' or 'car_rental' or 'vaccines' or 'international_visa' or 'translate' or 'carry_on' or 'book_flight' or 'timezone' or 'flight_status' or 'lost_luggage' or 'book_hotel' or 'plug_type' or 'travel_alert' or 'travel_notification' or 'travel_suggestion'"
    prompt = (f"{description}\n"
            f"Question: {text1} (Domain: {label_names[label1]})\n"
            f"Question: {text2} (Domain: {label_names[label2]})\n"
            f"Question:")
    return prompt

def get_prompt_ood(text1, label1, text2, label2, label_names):
    # define a function that takes as input two samples and generates the prompt
    # that we should pass to the GPT-3 language model for completion.
    #description = "Generate new OOD-questions for a bot, that have a completely different domain then the ID-question."
    description = "Generate new OOD-questions for a bot, that have a completely different domain then the ID-question. Each item in the following list contains a chatbot question and its Sub-Domain. Sub-Domain is one of 'travel, 'exchange_rate' or 'car_rental' or 'vaccines' or 'international_visa' or 'translate' or 'carry_on' or 'book_flight' or 'timezone' or 'flight_status' or 'lost_luggage' or 'book_hotel' or 'plug_type' or 'travel_alert' or 'travel_notification' or 'travel_suggestion'"
    prompt = (f"{description}\n"
            f"ID-Question: {text1} (Domain: {label_names[label1]})\n"
            f"OOD-Question: {'What is love?'} (Domain: {'football'})\n"
            f"ID-Question: {text2} (Domain: {label_names[label1]})\n"
            f"OOD-Question:")
    return prompt

def get_prompt_ood_2(texts, labels, label_names):
    # define a function that takes as input two samples and generates the prompt
    # that we should pass to the GPT-3 language model for completion.
    description = "Generate new OOD-questions for a bot, that have a completely different domain then the ID-question."
    #description = "Each item in the following list contains a chatbot question and its Sub-Domain. Sub-Domain is one of 'travel, 'exchange_rate' or 'car_rental' or 'vaccines' or 'international_visa' or 'translate' or 'carry_on' or 'book_flight' or 'timezone' or 'flight_status' or 'lost_luggage' or 'book_hotel' or 'plug_type' or 'travel_alert' or 'travel_notification' or 'travel_suggestion'"
    prompt = (f"{description}\n"
            f"ID-Question: {texts[0]} (Domain: {label_names[labels[0]]})\n"
            f"OOD-Question: {'How long does a football match last?'} (Domain: {'football'})\n"
            f"ID-Question: {texts[1]} (Domain: {label_names[labels[1]]})\n"
            f"OOD-Question: {'Is this your business?'} (Domain: {'business'})\n"
            f"ID-Question: {texts[2]} (Domain: {label_names[labels[2]]})\n"
            f"OOD-Question: {'Do you really have to take the pills?'} (Domain: {'medicine'})\n"
            f"ID-Question: {texts[3]} (Domain: {label_names[labels[3]]})\n"
            f"OOD-Question: {'Which is the fourth planet of the solar system?'} (Domain: {'astronomy'})\n"
            f"ID-Question: {texts[4]} (Domain: {label_names[labels[4]]})\n"
            f"OOD-Question: {'how long do cows live?'} (Domain: {'animals'})\n"
            f"ID-Question: {texts[5]} (Domain: {label_names[labels[5]]})\n"
            f"OOD-Question: {'How long does a wedding ceremony last?'} (Domain: {'marriage'})\n"
            f"ID-Question: {texts[6]} (Domain: {label_names[labels[6]]})\n"
            f"OOD-Question: {'where did abraham lincoln live?'} (Domain: {'celebrity'})\n"
            f"ID-Question: {texts[7]} (Domain: {label_names[labels[7]]})\n"
            f"OOD-Question: {'what years has korea been at war?'} (Domain: {'history'})\n"
            f"ID-Question: {texts[8]} (Domain: {label_names[labels[8]]})\n"
            f"OOD-Question: {'which piano is best for classical?'} (Domain: {'music'})\n"
            f"ID-Question: {texts[9]} (Domain: {label_names[labels[9]]})\n"
            f"OOD-Question: {'what team does eli mannign play for?'} (Domain: {'sports'})\n"
            f"ID-Question: {texts[10]} (Domain: {label_names[labels[10]]})\n"
            f"OOD-Question: {'how many oscars did star wars films win?'} (Domain: {'movies'})\n"
            f"ID-Question: {texts[11]} (Domain: {label_names[labels[11]]})\n"
            f"OOD-Question: {'why do males want to be alpha?'} (Domain: {'humans'})\n"
            f"ID-Question: {texts[12]} (Domain: {label_names[labels[12]]})\n"
            f"OOD-Question: {'how expensive is an apple share?'} (Domain: {'shares'})\n"
            f"ID-Question: {texts[13]} (Domain: {label_names[labels[13]]})\n"
            f"OOD-Question: {'how many sides are in a hexagon?'} (Domain: {'math'})\n"
            f"ID-Question: {texts[14]} (Domain: {label_names[labels[14]]})\n"
            f"OOD-Question:")
    return prompt

def get_label_excludes(labels):
    ret_labels = []
    for label in labels:
        if "_" in label:
            s1 = label.split("_")[0]
            s2 = label.split("_")[1]
            ret_labels.append(s1)
            ret_labels.append(s2)
            ret_labels.append(s2 + "_" + s1)

        ret_labels.append(label)

    return ret_labels



# synthetic_gpt3_10_ds = load_from_disk('./data/id_augm/gpt-3/10')
# synthetic_gpt3_50_ds = load_from_disk('./data/gpt-3/50')
# synthetic_gpt3_100_ds = load_from_disk('./data/gpt-3/100')
# synthetic_gpt3_200_ds = load_from_disk('./data/gpt-3/200')


if __name__ == "__main__":
    id_data_augm()