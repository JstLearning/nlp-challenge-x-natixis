import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np

from utils import save_model, save_results

import gc

from sklearn.metrics import f1_score, log_loss
from tqdm import tqdm

import datetime


def evaluate(model, val_loader, config, device, name="", epoch=0):
    criterion = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    preds = []
    labels = []
    output_proba = []
    method = config["method"]
    with torch.no_grad():
        total_loss = 0
        total_entries = 0
        correct = 0
        model.eval()
        with tqdm(val_loader, unit="batch") as tepoch:
            for batch in tepoch:
                # tqdm desc
                tepoch.set_description("Evaluation")

                if method is None:
                    X_ind, y = batch
                    X_ind = torch.Tensor(X_ind).float().to(device)
                    y = torch.Tensor(y).float().to(device)
                    
                    X_text = None
                    X_mask = None
                else:
                    X_ind = batch["X_ind"].to(device)
                    y = batch["label"].to(device)

                    if config["separate"]:
                        X_ecb = batch["X_ecb"].to(device)
                        X_ecb_att = batch["X_ecb_mask"].to(device)
                        X_fed = batch["X_fed"].to(device)
                        X_fed_att = batch["X_fed_mask"].to(device)

                        X_text = (X_ecb, X_fed)
                        X_mask = (X_ecb_att, X_fed_att)
                    else:
                        X_text = (batch["X_text"].to(device),)
                        X_mask = (batch["X_mask"].to(device),)

                output = model(X_text, X_mask, X_ind)
                # print(output)
                loss = criterion(output, y)

                output = sigmoid(output)
                output_proba.append(output.numpy(force=True))

                # Computing predictions
                batch_size_ = y.size(0)
                output = output.round()
                correct += (output == y).sum().item()
                total_loss += loss.item() * batch_size_
                total_entries += batch_size_

                preds.append(output.numpy(force=True))
                labels.append(y.numpy(force=True))

                # del X_ecb, X_ecb_att, X_fed, X_fed_att, X_ind, y, batch
                gc.collect()
                torch.cuda.empty_cache()
                tepoch.set_postfix(loss=total_loss/total_entries,
                                   accuracy=100. * correct/total_entries)
    eval_loss = total_loss/total_entries
    eval_accu = 100. * correct/total_entries

    PATH_OUTPUTS = "outputs/"
    np.savez(PATH_OUTPUTS + "outputs.npz", np.concatenate(output_proba), np.concatenate(labels), np.concatenate(preds), dtype=object)

    # Compute F1-score as well.
    output_proba = np.concatenate(output_proba)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    eval_loss = log_loss(labels, output_proba)
    eval_f1 = f1_score(labels, preds)
    save_results(output_proba, preds, labels, eval_loss, name, epoch)
    return eval_loss, eval_accu, eval_f1


def train(model, train_loader, val_loader, config,
          device, max_epochs=25, eval_every=1, name="",
          train_loss_history=[], starting_epoch=1):
    
    optimizer = Adam(model.parameters(),
                     betas=(0.9, 0.98),
                     lr=config["learning_rate"],
                     weight_decay=config["weight_decay"])
    if config["preload"]:
        state = torch.load(config["preload"])
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        starting_epoch = state['epoch']
        train_loss_history = state['train_loss_history']

    scheduler_step = config["scheduler_step"]
    early_stopping = config["early_stopping"]
    if scheduler_step > 0:
        if not config["preload"]:
            scheduler_ratio = config["scheduler_ratio"]
            scheduler_last_epoch = config["scheduler_last_epoch"]
            lr_min = config["learning_rate_min"]
            scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_last_epoch, eta_min=lr_min)
        else:
            scheduler.load_state_dict(state['scheduler'])
    else:
        scheduler=None
        
    criterion = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    best_accu = 0

    eval_losses = []
    eval_accus = []
    eval_f1s = []

    patience = 0
    my_patience = 5

    method = config["method"]

    for epoch in range(starting_epoch, max_epochs+1):
        total_loss = 0
        total_entries = 0
        correct = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch in tepoch:
                # Get inputs
                if method is None:
                    X_ind, y_ = batch
                    X_ind = torch.Tensor(X_ind).float().to(device)
                    y_ = torch.Tensor(y_).float().to(device)
                    
                    X_text = None
                    X_mask = None
                else:
                    X_ind = batch["X_ind"].to(device)
                    y_ = batch["label"].to(device)

                    if config["separate"]:
                        X_ecb = batch["X_ecb"].to(device)
                        X_ecb_att = batch["X_ecb_mask"].to(device)
                        X_fed = batch["X_fed"].to(device)
                        X_fed_att = batch["X_fed_mask"].to(device)

                        X_text = (X_ecb, X_fed)
                        X_mask = (X_ecb_att, X_fed_att)
                    else:
                        X_text = (batch["X_text"].to(device),)
                        X_mask = (batch["X_mask"].to(device),)
                
                # Compute output
                output = model(X_text, X_mask, X_ind)
                # print(output)

                # Compute loss
                loss = criterion(output, y_)
                
                # Update model
                loss.backward()
                optimizer.step()

                # Computing predictions

                ## Batch loss
                batch_loss = loss.item()

                # Accuracy computation
                output_proba = sigmoid(output)
                batch_size_ = y_.size(0)
                preds = output_proba.round()
                correct += (preds == y_).sum().item()
                ## Total loss with no reduction
                total_loss += loss.item() * batch_size_
                total_entries += batch_size_
                tepoch.set_postfix(loss=total_loss/total_entries,
                                    accuracy=100. * correct/total_entries,
                                    batch_loss=batch_loss)
        if (not scheduler is None) and epoch < scheduler_last_epoch:
            scheduler.step()
        train_loss_history.append(total_loss/total_entries)

        # Evaluation
        if epoch % eval_every == 0:
            eval_loss, eval_accu, eval_f1 = evaluate(
                model, val_loader, config, device, name, epoch)
            eval_losses.append(eval_loss)
            eval_accus.append(eval_accu)
            eval_f1s.append(eval_f1)

            if eval_accu > best_accu:
                best_accu = eval_accu
                # Save model
                save_model(model, name, optimizer, scheduler, epoch, train_loss_history, config)
                patience = 0
            else:
                patience += 1
                if patience >= my_patience and early_stopping:
                    print(f"Ran out of patience at epoch {epoch}.")
                    return eval_losses, eval_accus, eval_f1s
    
    return eval_losses, eval_accus, eval_f1s