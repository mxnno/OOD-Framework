#source https://github.com/ml6team/quick-tips/blob/main/nlp/2021_11_25_augmentation_lm/nlp_augmentation_lm.ipynb
import random
from datasets import load_from_disk, Dataset, ClassLabel

import json
import random
import requests
from data import preprocess_data
from model import  set_model
from utils.args import get_args
from utils.utils import get_labels


def id_data_augm():


    n = 20
    
    #get args
    args = get_args()

    #Load Model
    print("Load model...")
    _ , _ , tokenizer = set_model(args)

    #Preprocess Data
    print("Preprocess Data...")
    train_dataset, _ , _ = preprocess_data(args, tokenizer,no_Dataloader=True)

    label_name, label_id = get_labels(args)
    mapping = ClassLabel(names=label_name)

    # define the number of synthetic samples to generate
    new_texts = []
    new_labels = []
    api_key = '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'
    headers = {'Authorization' : 'Bearer ' + api_key,
                'Content-type':'application/json', 
                'Accept':'application/json'}

    iter = 0
    while iter < n:
        # select two random samples from training set
        text1, label1, text2, label2 = get_two_random_samples(train_dataset)
        # create the prompt
        prompt = get_prompt(text1, label1, text2, label2, mapping)
        # send a post request to gpt-3 using the prompt
        response = requests.post('https://api.openai.com/v1/engines/davinci/completions',
                                    headers=headers,
                                    data = json.dumps({"prompt": prompt, 
                                                    "max_tokens": 30,
                                                    "temperature": 0.9,
                                                    "top_p": 0.95}))

        # get response and extract the generated text and label
        # the generated output will be in the form "<text> (Sentiment: <label>)"
        data = response.json()['choices'][0]['text'].split('\n')[0].split('(Sentiment:')

        if len(data) < 2:
            # the format of the response is invalid
            continue

        text = data[0]
        label = data[1].split(')')[0].strip()

        if label not in label_name:
            # the format of the response is invalid
            continue

        new_texts.append(text)
        new_labels.append(mapping.str2int(label))
        iter += 1

        # define the synthetic dataset and save it to disk so as to prevent sending 
        # many api requests
        synthetic_ds = Dataset.from_dict({'text': new_texts, 'intent': new_labels})
        synthetic_ds.save_to_disk('./data/id_augm/gpt-3/' + str(n))



def get_two_random_samples(train_dataset):
    # define a function that returns two random samples from the train set.
    s1, s2 = random.sample(range(0, len(train_dataset)), 2)
    return train_dataset['text'][s1], train_dataset['intent'][s1], train_dataset['text'][s2], train_dataset['intent'][s2]

def get_prompt(text1, label1, text2, label2, mapping):
    # define a function that takes as input two samples and generates the prompt
    # that we should pass to the GPT-3 language model for completion.
    description = "Each item in the following list contains a chatbot question and its sub-domain. Sub-Domain is one of 'exchange_rate' or 'car_rental' or 'vaccines' or 'international_visa' or 'translate' or 'carry_on' or 'book_flight' or 'timezone' or 'flight_status' or 'lost_luggage' or 'book_hotel' or 'plug_type' or 'travel_alert' or 'travel_notification' or 'travel_suggestion'"
    prompt = (f"{description}\n"
            f"Question: {text1} (Domain: {mapping.int2str(label1)})\n"
            f"Question: {text2} (Domain: {mapping.int2str(label2)})\n"
            f"Question:")
    return prompt







synthetic_gpt3_10_ds = load_from_disk('./data/id_augm/gpt-3/10')
synthetic_gpt3_50_ds = load_from_disk('./data/gpt-3/50')
synthetic_gpt3_100_ds = load_from_disk('./data/gpt-3/100')
synthetic_gpt3_200_ds = load_from_disk('./data/gpt-3/200')


if __name__ == "__main__":
    id_data_augm()