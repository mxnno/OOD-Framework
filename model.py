import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaPreTrainedModel, RobertaModel, AutoModelForSequenceClassification
from sklearn.covariance import EmpiricalCovariance

class RobertaClassificationHead(nn.Module):
    #https://github.com/pytorch/fairseq/blob/a54021305d6b3c4c5959ac9395135f63202db8f1/fairseq/models/roberta/model.py#L394-L429
    #https://github.com/huggingface/transformers/blob/8f3ea7a1e1a85e80210b3d4423b674d9a61016ed/src/transformers/models/roberta/modeling_roberta.py#L1432
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
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

    def forward(self, input_ids=None, attention_mask=None, labels=None, onlyPooled=None):
        #input_ids, attention_mask, labels kommt von tokenized_datasets ins model:
        
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)

        if onlyPooled:
            return pooled 
        
        loss = None
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
        

    def compute_ood(self, input_ids=None, attention_mask=None, labels=None):

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits, pooled = self.classifier(sequence_output)

        ood_keys = None
        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]

        maha_score = []
        for c in self.all_classes:
            centered_pooled = pooled - self.class_mean[c].unsqueeze(0)
            ms = torch.diag(centered_pooled @ self.class_var @ centered_pooled.t())
            maha_score.append(ms)
        maha_score = torch.stack(maha_score, dim=-1)
        maha_score = maha_score.min(-1)[0]
        maha_score = -maha_score

        norm_pooled = F.normalize(pooled, dim=-1)
        cosine_score = norm_pooled @ self.norm_bank.t()
        cosine_score = cosine_score.max(-1)[0]

        energy_score = torch.logsumexp(logits, dim=-1)

        ood_keys = {
            'softmax': softmax_score.tolist(),
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'energy': energy_score.tolist(),
        }
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
        self.all_classes = list(set(self.label_bank.tolist()))
        self.class_mean = torch.zeros(max(self.all_classes) + 1, d).cuda()
        for c in self.all_classes:
            self.class_mean[c] = (self.bank[self.label_bank == c].mean(0))
        centered_bank = (self.bank - self.class_mean[self.label_bank]).detach().cpu().numpy()
        precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        self.class_var = torch.from_numpy(precision).float().cuda()
