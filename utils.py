from pathlib import Path
import torch
import json

PATH_OUTPUTS = Path(__name__).parent / "outputs"

def save_model(model, name, optimizer, scheduler, epoch, train_loss_history, config):
    PATH_MODEL = PATH_OUTPUTS / f"model_{name}"
    PATH_MODEL.mkdir(exist_ok=True)
    PATH_MODEL_EPOCH = PATH_MODEL / f"epoch_{epoch}"
    PATH_MODEL_EPOCH.mkdir(exist_ok=True)
    if config["scheduler_step"] > 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                }, 
                PATH_MODEL_EPOCH / f"model_{name}_epoch_{epoch}.pt")
    else:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                }, 
                PATH_MODEL_EPOCH / f"model_{name}_epoch_{epoch}.pt")
    with open(PATH_MODEL_EPOCH / f'model_{name}_config.json', 'w') as f:
        json.dump(config, f)

def save_results(outputs_proba, outputs, targets, logloss, name, epoch):
    PATH_MODEL = PATH_OUTPUTS / f"model_{name}"
    PATH_MODEL.mkdir(exist_ok=True)
    PATH_MODEL_EPOCH = PATH_MODEL / f"epoch_{epoch}"
    PATH_MODEL_EPOCH.mkdir(exist_ok=True)
    with open(PATH_MODEL_EPOCH / f'model_{name}_results.json', 'w') as f:
        json.dump({
            "name": str(name),
            "outputs_proba": outputs_proba.astype(str).tolist(),
            "outputs": outputs.astype(str).tolist(),
            "targets": targets.astype(str).tolist(),
            "logloss": str(logloss),
            "epoch": epoch
        }, f)
