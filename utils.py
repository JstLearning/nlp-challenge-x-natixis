from pathlib import Path
import torch
import json

PATH_OUTPUTS = Path(__name__).parent / "outputs"

def save_model(model, name, optimizer, epoch, train_loss_history):
    PATH_MODEL = PATH_OUTPUTS / f"model_{name}"
    PATH_MODEL.mkdir(exist_ok=True)
    PATH_MODEL_EPOCH = PATH_MODEL / f"epoch_{epoch}"
    PATH_MODEL_EPOCH.mkdir(exist_ok=True)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss_history': train_loss_history
            }, 
            PATH_MODEL_EPOCH / f"model_{name}_epoch_{epoch}.pt")

def save_results(outputs_proba, outputs, targets, logloss, name, epoch):
    PATH_MODEL = PATH_OUTPUTS / f"model_{name}"
    PATH_MODEL.mkdir(exist_ok=True)
    PATH_MODEL_EPOCH = PATH_MODEL / f"epoch_{epoch}"
    PATH_MODEL_EPOCH.mkdir(exist_ok=True)
    with open(PATH_MODEL_EPOCH / f'model_{name}_results.json', 'w') as f:
        json.dump({
            "name": str(name),
            "outputs_proba": list(outputs_proba),
            "outputs": list(outputs),
            "targets": list(targets),
            "logloss": logloss,
            "epoch": epoch
        }, f)


    
