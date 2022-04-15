from lib2to3.pgen2 import token
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaPreTrainedModel, RobertaForMaskedLM,  RobertaModel, RobertaConfig, RobertaTokenizer, RobertaTokenizerFast
from sklearn.covariance import EmpiricalCovariance
from utils.utils_ADB import euclidean_metric
from utils.utils import get_num_labels
from scipy.stats import entropy


def set_model(args):

    num_labels = get_num_labels(args)

    if args.model_ID == 1:

        
        if args.model_name_or_path.startswith("roberta"):
            #erst IMLM Finetuning mit Roberta + MaskedLM
            config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
            config.gradient_checkpointing = True
            config.alpha = args.alpha
            config.loss = args.loss
            model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
            tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name_or_path)
        else:
            #damm BCAD Finetuning mit dem IMLM-finegetuned model, das extra abgespeichert wird
            config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels)
            config.gradient_checkpointing = True
            config.alpha = args.alpha
            config.loss = args.loss
            model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model.to(args.device)

        
    else:
        if args.model_ID == 8:
            config = RobertaConfig.from_pretrained('roberta-base', num_labels=num_labels)
        else:
            config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)

    return model, config, tokenizer
#####################################################################################################################################################

class RobertaClassificationHead(nn.Module):
    #https://github.com/pytorch/fairseq/blob/a54021305d6b3c4c5959ac9395135f63202db8f1/fairseq/models/roberta/model.py#L394-L429
    #https://github.com/huggingface/transformers/blob/8f3ea7a1e1a85e80210b3d4423b674d9a61016ed/src/transformers/models/roberta/modeling_roberta.py#L1432
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        #feature  -> sequence_output/last_hidden_state = (batch_size, sequence_length, hidden_size)
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        #Roberta does not have a pooler layer (like Bert for instance) since the pretraining objective does not contain a classification task
        #BertPooler (which is just dense + tanh)
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        x = self.dense(x)
        x = pooled = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, pooled


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    #https://github.com/huggingface/transformers/blob/8f3ea7a1e1a85e80210b3d4423b674d9a61016ed/src/transformers/models/roberta/modeling_roberta.py#L1163
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()
        #self.post_init()
        #die beiden unternen nr für Test -> können weg
        self.all_logits = []
        self.all_pool = []

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, onlyPooled=None):

        #input_ids, attention_mask, labels kommt von tokenized_datasets ins model:
        
        #output type: BaseModelOutputWithPoolingAndCrossAttentions (aber ohen Pooling, da oben add_pooling_layer=False)
        #(last_hidden_state=tensor([...]), pooler_output=None, hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None)
        if token_type_ids is not None:
            #für DNNC brauchen wir token_type_ids (Satz1 vs Satz2)
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
        #sequence_output = last_hidden_state embeddings =  (batch_size, sequence_length, hidden_size) [8, 512, 768]
        sequence_output = outputs[0]
        #pooled output: embeddings of [CLS] Token
        #logits: [batch_size, num_labels] -> [-4.22, 4.354], [-2.22, 3.4]... 
        logits, pooled = self.classifier(sequence_output)

        if onlyPooled:
            return pooled 
        
        loss = None
        #loss nur wenn label
        if labels is not None:

            #Standard loss
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            if self.config.loss == 'margin-contrastive':
                # ID 2
                dist = ((pooled.unsqueeze(1) - pooled.unsqueeze(0)) ** 2).mean(-1)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                max_dist = (dist * mask).max()
                cos_loss = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()

                loss = loss + self.config.alpha * cos_loss
                output = (logits,) + outputs[2:]
                output = output + (pooled,)
                return ((loss, cos_loss) + output) if loss is not None else output

            elif self.config.loss == 'similarity-contrastive':
                # ID 2
                norm_pooled = F.normalize(pooled, dim=-1)
                cosine_score = torch.exp(norm_pooled @ norm_pooled.t() / 0.3)
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
                mask = mask - torch.diag(torch.diag(mask))
                cos_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
                cos_loss = -torch.log(cos_loss + 1e-5)
                cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
                cos_loss = cos_loss.mean()

                loss = loss + self.config.alpha * cos_loss
                output = (logits,) + outputs[2:]
                output = output + (pooled,)
                return ((loss, cos_loss) + output) if loss is not None else output
            else:
                pass
                
        output = (logits,) + outputs[2:]
        output = output + (pooled,)
        return ((loss, None) + output) if loss is not None else output
        #return (loss, None/cos_loss, logits, outputs[2:], pooled,)
        

    def compute_ood(self, input_ids=None, attention_mask=None, labels=None, centroids=None, delta=None, test=None):

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)
        

        self.all_logits.append(logits)
        self.all_pool.append(pooled)


        ood_keys = None
        #Softmax
        #max: Returns the maximum value of all elements in the input tensor (max, max_indices)
        #softmax_score(Scores der prediction für alle Klassen -> max(-1) gibt den Max Wert aus)
        #max_indices = Klasse, die den Max Wert hat
        softmax_score, softmax_idx= F.softmax(logits, dim=-1).max(-1)
  
        
        #Maha
        maha_score = []
        for c in self.all_classes:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score = maha_score.min(-1)[0]
        maha_score = -maha_score

        #Cosine
        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]

        #Energy
        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
        }

        #ADB
        euc_dis = []
        if centroids is not None:
            logits_adb = euclidean_metric(pooled, centroids)
            probs, preds = F.softmax(logits_adb.detach(), dim = 1).max(dim = 1)
            euc_dis = torch.norm(pooled - centroids[preds], 2, 1).view(-1)
            #data.unseen_token_id = num_labels -> bei 5 labels -> 0-4: ID -> 5: OOD 
            preds[euc_dis >= delta[preds]] = self.num_labels
            #return preds
            #ood_keys['adb'] = euc_dis.tolist()
            ood_keys['adb'] = preds
            #preds wird dann in cm gepackt un dan noch mit F-measure berechnet (https://github.com/thuiar/Adaptive-Decision-Boundary/blob/727dfca95cd254cbfa165a43517304403f6a59ea/util.py#L64)
        
        if test is not None:
            return ood_keys, logits, softmax_idx
        return ood_keys

    def prepare_ood(self, dataloader=None):
        self.bank = None
        self.label_bank = None
        for batch in dataloader:
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch['labels']
            #forward ohne labels
            outputs = self.roberta(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
            sequence_output = outputs[0]
            logits, pooled = self.classifier(sequence_output)
            if self.bank is None:
                self.bank = pooled.clone().detach()
                self.label_bank = labels.clone().detach()

            else:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
                self.bank = torch.cat([bank, self.bank], dim=0)

                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)

        self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = self.bank.size()
        #all_classes = liste mit label [0,1,2,3,4...14]
        self.all_classes = list(set(self.label_bank.tolist()))
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()
        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()

        torch.save(self.norm_bank, 'norm_bank.pt') 
        torch.save(self.class_var, 'class_var.pt') # torch.Size([768, 768])
        torch.save(self.class_mean, 'class_mean.pt') # torch.Size([15, 768])







# Grundlagen KI
# Single Node: output = activation((x MatMul W) + B)
# Dense Layer: Fully-Connected Layer
# Activation Function: wandelt die Aktivierung der einzelnen Neuronen um bzw. grenzt die Aktivierung ein. z.B. von (0,1) in (0.1 .. 0.9) oder relu max(0,x) -> output = activation(Sum(W))
# Bsp. Softmax: Für Output Layer:  for multi-class classification problems where class membership is required on more than two class labels
# Logits: Output NN (unnormalized predictions ) and Inputs to Softmax


#NLP: 

#Sequence/Text Classifcation: task of classifying sequences according to a given number of classes -> pooled output = cls Token (siehe classifier)
#Token Classification: task in which a label is assigned to some tokens in a text z.B. Named Entity Recognition (NER) -> last layer Output (batch_size, sequence_length, config.vocab_size)
#Embeddings = Tokenizer.input_ids = Dictonary/Lookup Table für Wörter in Zahlen. Bei Bert z.B. auf 30k beschränkt -> "embeddings" = 'em', '##bed', '##ding' -> em = 1023, bed = 2152

#NLI: In NLI the model determines the relationship between two given texts. Concretely, the model takes a premise and a hypothesis and returns a class that can either be:
#   entailment, which means the hypothesis is true. 
#   contraction, which means the hypothesis is false.
#   neutral, which means there's no relation between the hypothesis and the premise.