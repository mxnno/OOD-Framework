import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ID", type=int)
    parser.add_argument("--task", type=str, choices=['finetune','ood_detection'])
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--dataset", type=str, default="clinc150")
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--ood_data", default="full", type=str, choices=['full','zero', 'augm'])
    parser.add_argument("--id_data", default="full", type=str)
    parser.add_argument("--few_shot", default="100", type=int)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--save_path", default="drive", type=str)
    parser.add_argument("--wandb", type=str, default="log")
    parser.add_argument("--tpu", type=str, default="gpu")

    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss_std", type=str, choices=['ce', 'lmcl'], default='ce')
    parser.add_argument("--loss", type=str, choices=['margin-contrastive', 'similarity-contrastive', 'similarity-contrastive-augm', 'default'], default='default')
    #für ID 8:
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument('--label_smoothing',type = float, default = 0.1, help = 'Coefficient for label smoothing (default: 0.1, if 0.0, no label smoothing)')
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,type=float)
    
    return parser.parse_args()