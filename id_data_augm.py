#source https://github.com/ml6team/quick-tips/blob/main/nlp/2021_11_25_augmentation_lm/nlp_augmentation_lm.ipynb


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


    n = 2
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
            prompt = "give me chatbot questions, that do not belong to the following domains: 'exchange_rate', 'car_rental', 'vaccines', 'international_visa', 'translate', 'carry_on', 'book_flight', 'timezone', 'flight_status', 'lost_luggage', 'book_hotel', 'plug_type', 'travel_alert', 'travel_notification' or 'travel_suggestion' ?"
            #prompt = "can you give me an example of a chatbot question that is the same format like this question: 'How much is a dollar in euro?' but has a completely different context/domain?"
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

            if ood_augm:
                if label == label_name:
                # the format of the response is invalid
                    continue
            else:
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

    if ood_augm:
        full_aug_ds.save_to_disk('./data/id_augm/gpt-3/full_ood_augm/' )
        full_aug_ds.to_csv("ood_augm_ds.csv")
    else:
        full_aug_ds.save_to_disk('./data/id_augm/gpt-3/full_id_augm/' )
        full_aug_ds.to_csv("id_augm_ds.csv")





def get_two_random_samples(train_dataset):
    # define a function that returns two random samples from the train set.
    s1, s2 = random.sample(range(0, len(train_dataset)), 2)
    return train_dataset['text'][s1], train_dataset['intent'][s1], train_dataset['text'][s2], train_dataset['intent'][s2]

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
    description = "Generate new OOD-questions for a bot, that have a completely different domain then the ID-question."
    #description = "Each item in the following list contains a chatbot question and its Sub-Domain. Sub-Domain is one of 'exchange_rate' or 'car_rental' or 'vaccines' or 'international_visa' or 'translate' or 'carry_on' or 'book_flight' or 'timezone' or 'flight_status' or 'lost_luggage' or 'book_hotel' or 'plug_type' or 'travel_alert' or 'travel_notification' or 'travel_suggestion'"
    prompt = (f"{description}\n"
            f"ID-Question: {text1} (Domain: {label_names[label1]})\n"
            f"OOD-Question: {'How long does a football match last?'} (Domain: {'footbal'})\n"
            f"ID-Question: {text2} (Domain: {label_names[label1]})\n"
            f"OOD-Question:")
    return prompt







# synthetic_gpt3_10_ds = load_from_disk('./data/id_augm/gpt-3/10')
# synthetic_gpt3_50_ds = load_from_disk('./data/gpt-3/50')
# synthetic_gpt3_100_ds = load_from_disk('./data/gpt-3/100')
# synthetic_gpt3_200_ds = load_from_disk('./data/gpt-3/200')


if __name__ == "__main__":
    id_data_augm()