from tqdm import tqdm
from transformers.optimization import get_scheduler
import torch.nn.functional as F
import torch
import wandb
import warnings
import numpy as np
from evaluation import evaluate
from optimizer import get_optimizer_2
from utils.utils_ADB import BoundaryLoss
warnings.filterwarnings("ignore")


def finetune(args, model, train_dataloader, dev_dataloader):

    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = get_optimizer_2(args, model)
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
        #ToDo: bestes Model speichern
        wandb.log(results, step=num_steps)

    if args.save_path:
        model.save_pretrained(args.save_path)
        print("Model saved at: " + args.save_path)

    return model


def finetune_ADB(args, model, train_dataloader, dev_dataloader):
    
    #args überarbeiten

    criterion_boundary = BoundaryLoss(num_labels = model.num_labels, feat_dim = args.feat_dim)
    delta = F.softplus(criterion_boundary.delta)

    #evtl parser.add_argument("--lr_boundary", type=float, default=0.05, help="The learning rate of the decision boundary.")
    optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = 0.05)

    def centroids_cal(args, train_dataloader):
        centroids = torch.zeros(model.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(args.device)

        with torch.set_grad_enabled(False):
            for batch in train_dataloader:
                #batch = {key: value.to(args.device) for key, value in batch.items()}
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, attention_mask, labels_ids = batch
                features = model(input_ids, attention_mask, labels_ids, onlyPooled=True)
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]
                
        total_labels = total_labels.cpu().numpy()

        def class_count(labels):
            class_data_num = []
            for l in np.unique(labels):
                num = len(labels[labels == l])
                class_data_num.append(num)
            return class_data_num


        centroids /= torch.tensor(class_count(total_labels)).float().unsqueeze(1).cuda()
        
        return centroids


    centroids = centroids_cal(args, train_dataloader)

    wait = 0
    best_delta, best_centroids = None, None

    num_steps = 0

    for epoch in int(args.num_train_epochs):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, label_ids = batch
            with torch.set_grad_enabled(True):
                features = model(input_ids, input_mask, onlyPooled=True)
                loss, delta = criterion_boundary(features, centroids, label_ids)

                optimizer.zero_grad()
                loss.backward()
                num_steps += 1
                optimizer.step()
                
                tr_loss += loss.item()
                
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

    #     delta_points.append(delta)
        
    #     # if epoch <= 20:
    #     #     plot_curve(self.delta_points)

    #     loss = tr_loss / nb_tr_steps
    #     print('train_loss',loss)
        
    #     eval_score = evaluation(args, data, mode="eval")
    #     print('eval_score',eval_score)
        
    #     if eval_score >= self.best_eval_score:
    #         wait = 0
    #         self.best_eval_score = eval_score
    #         best_delta = self.delta
    #         best_centroids = self.centroids
    #     else:
    #         wait += 1
    #         if wait >= args.wait_patient:
    #             break
    
    # delta = best_delta
    # centroids = best_centroids

    #EVTL IST DIE save_results METHODE INTERESSANT (PLOTS!) -> ist aber nicht hier -> im Orginalcode

    results = evaluate(args, model, dev_dataloader, tag="dev")
    #ToDo: bestes Model speichern
    wandb.log(results, step=num_steps)

    if args.save_path:
        model.save_pretrained(args.save_path)
        print("Model saved at: " + args.save_path)




