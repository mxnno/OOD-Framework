from tqdm import tqdm
from transformers.optimization import AdamW, get_scheduler
import wandb
import warnings
from evaluation import evaluate
warnings.filterwarnings("ignore")


def finetune(args, model, train_dataloader, dev_dataloader):

    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_scheduler("linear", optimizer=optimizer,num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    num_steps = 0
    model.train()
    for epoch in range(int(args.num_train_epochs)):
        print("Epoche: " + str(epoch))
        #model.zero_grad()
        for batch in tqdm(train_dataloader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            wandb.log({'loss': loss.item()}, step=num_steps)
            if cos_loss:
                wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)

        results = evaluate(args, model, dev_dataloader, tag="dev")
        wandb.log(results, step=num_steps)

    if args.save_path:
        #Model abspeichern
        pass