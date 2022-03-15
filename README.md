# OOD-Framework

## Included OOD-Detection methods
* [Contrastive Out-of-Distribution Detection for Pretrained Transformers](https://arxiv.org/abs/2104.08812)
* ...

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* datasets
* wandb
* tqdm
* scikit-learn

## Dataset
[clinc150](https://github.com/clinc/oos-eval): The 'few-shot' parameter can be used to reduce the training data.

## Training and Evaluation

Finetune the model with the following command:

```bash
python run.py --task finetune --model_name_or_path roberta-base --model_ID 0 --ood_data zero --id_data full --few_shot 10
```

Evaluate the model OOD-Detection with the following command:

```bash
python run.py --task ood_detection --model_name_or_path roberta-base --model_ID 0 --ood_data zero --id_data full --few_shot 10
```
