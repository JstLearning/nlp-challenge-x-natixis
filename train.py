import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

import gc

from sklearn.metrics import f1_score
from tqdm import tqdm


def evaluate(model, val_loader, config, device):
    criterion = nn.BCELoss()
    preds = []
    labels = []
    with torch.no_grad():
        total_loss = 0
        total_entries = 0
        correct = 0
        model.eval()
        with tqdm(val_loader, unit="batch") as tepoch:
            for batch in tepoch:
                # tqdm desc
                tepoch.set_description("Evaluation")

                X_ind = batch["X_ind"].to(device)
                y = batch["label"]

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
                loss = criterion(output, y)

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

    # Compute F1-score as well.
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    eval_f1 = f1_score(labels, preds)
    return eval_loss, eval_accu, eval_f1


def train(model, train_loader, val_loader, config,
          device, max_epochs=25, eval_every=1, name=""):
    optimizer = Adam(model.parameters(),
                     lr=config["learning_rate"],
                     weight_decay=config["weight_decay"])
    criterion = nn.BCELoss()

    best_accu = 0

    eval_losses = []
    eval_accus = []
    eval_f1s = []

    patience = 0
    my_patience = 2

    method = config["method"]

    for epoch in range(max_epochs):
        total_loss = 0
        total_entries = 0
        correct = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                # tqdm desc
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

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
                loss.backward()
                optimizer.step()

                # Computing predictions
                batch_size_ = y.size(0)
                output = output.round()
                correct += (output == y).sum().item()
                total_loss += loss.item() * batch_size_
                total_entries += batch_size_

                # del X_ecb, X_ecb_att, X_fed, X_fed_att, X_ind, y, batch
                gc.collect()
                torch.cuda.empty_cache()
                tepoch.set_postfix(loss=total_loss/total_entries,
                                   accuracy=100. * correct/total_entries)
        del X_ecb, X_ecb_att, X_fed, X_fed_att, X_ind, y, batch
        gc.collect()
        torch.cuda.empty_cache()
        # Evaluation
        if epoch % eval_every == 0:
            eval_loss, eval_accu, eval_f1 = evaluate(
                model, val_loader, config, device)
            eval_losses.append(eval_loss)
            eval_accus.append(eval_accu)
            eval_f1s.append(eval_f1)

            if eval_accu > best_accu:
                best_accu = eval_accu
                # Save model
                torch.save(model.state_dict(),
                           f"./model/weights/model_{name}_{epoch}_epochs.pt")
                patience = 0
            else:
                patience += 1
                if patience >= my_patience:
                    return eval_losses, eval_accus, eval_f1s
                

    return eval_losses, eval_accus, eval_f1s
    

def train_with_accumulation(model, train_loader, val_loader, config,
        device, acc_steps=2, max_epochs=25, eval_every=1, name=""):
    optimizer = Adam(model.parameters(),
                        lr=config["learning_rate"],
                        weight_decay=config["weight_decay"])
    criterion = nn.BCELoss()

    best_accu = 0

    eval_losses = []
    eval_accus = []
    eval_f1s = []

    patience = 0
    my_patience = 2

    method = config["method"]

    for epoch in range(max_epochs):
        total_loss = 0
        total_entries = 0
        correct = 0
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            acc_step = 0
            optimizer.zero_grad()
            for batch in tepoch:
                # tqdm desc
                tepoch.set_description(f"Epoch {epoch}")
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
                loss.backward()
                acc_step += 1
                if acc_step % acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    acc_step = 0

                # Computing predictions
                batch_size_ = y.size(0)
                output = output.round()
                correct += (output == y).sum().item()
                total_loss += loss.item() * batch_size_
                total_entries += batch_size_

                # del X_ecb, X_ecb_att, X_fed, X_fed_att, X_ind, y, batch
                gc.collect()
                torch.cuda.empty_cache()
                tepoch.set_postfix(loss=total_loss/total_entries,
                                    accuracy=100. * correct/total_entries)
        del X_ecb, X_ecb_att, X_fed, X_fed_att, X_ind, y, batch
        gc.collect()
        torch.cuda.empty_cache()
        # Evaluation
        if epoch % eval_every == 0:
            eval_loss, eval_accu, eval_f1 = evaluate(
                model, val_loader, config, device)
            eval_losses.append(eval_loss)
            eval_accus.append(eval_accu)
            eval_f1s.append(eval_f1)

            if eval_accu > best_accu:
                best_accu = eval_accu
                # Save model
                torch.save(model.state_dict(),
                            f"./model/weights/model_{name}_{epoch}_epochs.pt")
                patience = 0
            else:
                patience += 1
                if patience >= my_patience:
                    return eval_losses, eval_accus, eval_f1s
                

    return eval_losses, eval_accus, eval_f1s
