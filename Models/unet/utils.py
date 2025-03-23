import torch
import torchvision
from dataset import BrainTumorSegmentationDataset
from torch.utils.data import DataLoader
import os
from datetime import datetime
import matplotlib.pyplot as plt

CHECKPOINT_DIR = "checkpoints"

# def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
#     print("=> Saving checkpoint")
#     torch.save(state, filename)

def save_checkpoint(state):
    # Ensure the directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Get the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get the next available checkpoint ID
    existing_files = os.listdir(CHECKPOINT_DIR)
    checkpoint_ids = [
        int(f.split("_")[1]) for f in existing_files if f.startswith("checkpoint_")
    ]
    next_id = max(checkpoint_ids, default=0) + 1  # Start from 1 if empty

    # Create filename: checkpoint_<id>_<timestamp>.pth.tar
    checkpoint_filename = f"checkpoint_{next_id}_{timestamp}.pth.tar"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)

    # Save checkpoint
    torch.save(state, checkpoint_path)
    print(f"✅ Checkpoint saved: {checkpoint_filename}")

# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])

def load_checkpoint(model, optimizer, checkpoint_name=None):
    if checkpoint_name is None:
        # Get the list of all saved checkpoints
        files = sorted(os.listdir(CHECKPOINT_DIR), reverse=True)  # Sort newest first
        if not files:
            print("❌ No checkpoints found!")
            return
        checkpoint_name = files[0]  # Pick the latest checkpoint

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
            y = y.to(device).unsqueeze(1) # unsqueeze for 1 channel
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.4).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda", show_last_epoch=False
):
    new_folder = os.path.join(folder, "MatplotOutputs/")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    model.eval()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds >= 0.4).float()
        
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}-{timestamp}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1).to(torch.float32), f"{folder}/{idx}.png")

        if show_last_epoch:
            for i in range(x.size(0)):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                ax[0].imshow(x[i].cpu().squeeze(0), cmap='gray')
                ax[0].set_title('Input Image')

                ax[1].imshow(y[i].cpu().squeeze(), cmap='gray')
                ax[1].set_title('Ground Truth')

                ax[2].imshow(preds[i].cpu().squeeze(), cmap='gray')
                ax[2].set_title('Prediction')

                fig.savefig(f"{new_folder}/output-{idx}-{timestamp}.png")
                plt.show()
                plt.close(fig)

    model.train()