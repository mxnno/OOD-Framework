import csv
import json
from utils.utils import get_labels

def create_gold_json(args):

    labels_name, _ = get_labels(args)

    csv_list = ["/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/train.csv", "/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/validation.csv", "/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/test.csv"]
    key = ["train", "dev", "test"]
    # create a dictionary
    data = {}
    for i, csvPath in enumerate(csv_list):

        # Open a csv reader called DictReader
        with open(csvPath, encoding='utf-8') as csvf:
            csvReader = csv.reader(csvf, delimiter=',')
            
            inner_Dict = {}
            inner_Dict_list = []
            # Convert each row into a dictionary
            # and add it to data
            counter = 0
            for row in csvReader:

                inner_Dict = {}
                counter+= 1
                if counter == 1:
                    continue

                inner_Dict["guid"] = key[i] + "_" + str(counter-1)
                inner_Dict["split"] = key[i]
                inner_Dict["context"] = row[1]
                    
                if labels_name[int(row[2])] == "ood":
                    #oos
                    inner_Dict["oos_label"] = 1
                    inner_Dict["intent_label"] = -1
                    inner_Dict["label_text"] = "oos"
                else:
                    inner_Dict["oos_label"] = 0
                    inner_Dict["intent_label"] = int(row[2])
                    inner_Dict["label_text"] = labels_name[int(row[2])]
                
                inner_Dict["agent_text"] = ""

                inner_Dict_list.append(inner_Dict)

        data[key[i]] = inner_Dict_list

        
    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open("/content/drive/MyDrive/Masterarbeit/OOD-Methoden/GOLD/gold.json", 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
        
    # Driver Code

# {
#     "train": [
#       {
#         "guid": "train_1",
#         "split": "train",
#         "context": "tell me the weather report for half moon bay",
#         "oos_label": 0,
#         "intent_label": 3,
#         "label_text": "find_weather",
#         "agent_text": ""
#       },
#  {
#         "guid": "dev_4493",
#         "split": "dev",
#         "context": "is my yoga dvd downstairs?",
#         "oos_label": 1,
#         "intent_label": -1,
#         "label_text": "out_of_scope",
#         "agent_text": ""
#       },