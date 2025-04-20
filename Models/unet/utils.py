import torch
import torchvision
from dataset import BrainTumorSegmentationDataset
from torch.utils.data import DataLoader
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random as rand

rand.seed(55)

CHECKPOINT_DIR = "U_Net_checkpoints"
THRESHOLD = 0.4

def save_checkpoint(state):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    existing_files = os.listdir(CHECKPOINT_DIR)
    checkpoint_ids = [
        int(f.split("_")[3]) for f in existing_files if f.startswith("U_Net_checkpoint_")
    ]
    next_id = max(checkpoint_ids, default=0) + 1 

    checkpoint_filename = f"U_Net_checkpoint_{next_id}_{timestamp}.pth.tar"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)

    torch.save(state, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_filename}")

def load_checkpoint(model, optimizer, checkpoint_name=None):
    if checkpoint_name is None:
        files = sorted(os.listdir(CHECKPOINT_DIR), reverse=True)  
        if not files:
            print("❌ No checkpoints found!")
            return
        checkpoint_name = files[0]  

    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint '{checkpoint_name}' not found!")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    print(f"✅ Loaded checkpoint: {checkpoint_name}")


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=2,
    pin_memory=True,
):
    train_ds = BrainTumorSegmentationDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BrainTumorSegmentationDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) 
            preds = torch.sigmoid(model(x))
            preds = (preds >= THRESHOLD).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    val_acc= num_correct/num_pixels*100 
    val_dice= dice_score/len(loader)
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {val_acc : .2f}"
    )
    print(f"Dice score: {val_dice}")

    model.train()

    return val_acc, val_dice
 
def save_predictions_as_imgs(
    loader, 
    model, 
    checkpoint_filename, 
    train_losses, 
    val_accs, 
    val_dice_scores, 
    folder="U_Net_saved_images/", 
    device="cuda", 
    show_last_epoch=False, 
):
    folder  = os.path.join(folder, checkpoint_filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    new_folder = os.path.join(folder, "U_Net_MatplotOutputs/")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    model.eval()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for idx, (x, y) in enumerate(loader):

        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds >= THRESHOLD).float()
        
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}-{timestamp}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1).to(torch.float32), f"{folder}/{idx}.png"
        )
        torchvision.utils.save_image(
            x.squeeze(0).to(torch.float32), f"{folder}/original_{idx}.png"
        )

        if show_last_epoch:
            i = rand.randint(0, x.size(0)-1)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))

            ax[0].imshow(x[i].cpu().squeeze(0), cmap='gray')
            ax[0].set_title('Input Image')

            ax[1].imshow(y[i].cpu().squeeze(), cmap='gray')
            ax[1].set_title('Ground Truth')

            ax[2].imshow(preds[i].cpu().squeeze(), cmap='gray')
            ax[2].set_title('Prediction')

            fig.savefig(f"{new_folder}/output-0-batch-{idx}-{timestamp}.png")
            plt.show()
            plt.close(fig)
            
    if show_last_epoch:    
        epochs = range(1,len(train_losses)+1)
        plt.figure(figsize=(12,4))

        #loss
        plt.subplot(1,3,1)
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.xticks(epochs)
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        #acc
        plt.subplot(1,3,2)
        plt.plot(epochs, val_accs, label='Val Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.xticks(epochs)
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        
        #dice
        plt.subplot(1,3,3)
        plt.plot(epochs, val_dice_scores, label='Dice Score', color='green')
        plt.xlabel('Epoch')
        plt.xticks(epochs)
        plt.ylabel('Dice')
        plt.title('Dice Score')

        plt.tight_layout()
        plt.savefig(f"{folder}/results_metrics_{timestamp}.png")
        plt.show()
        plt.close()

    model.train()
    