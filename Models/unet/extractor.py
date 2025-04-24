import re
import matplotlib.pyplot as plt
import os
from datetime import datetime

def extract_metrics_from_log(log_file_path):
    train_losses = []
    val_accs = []
    val_dice_scores = []
    train_nasar = []
    
    # Regular expressions to match each metric
    loss_pattern = re.compile(r'loss=([\d.]+)')
    nasar_pattern = re.compile(r'NASAR: ([\d.]+)')
    acc_pattern = re.compile(r'acc\s+([\d.]+)')
    dice_pattern = re.compile(r'Dice score: ([\d.]+)')
    
    with open(log_file_path, 'r') as file:
        for line in file:
            # Check for each metric in each line
            if 'loss=' in line:
                loss_match = loss_pattern.search(line)
                if loss_match:
                    train_losses.append(float(loss_match.group(1)))
            
            elif 'NASAR:' in line:
                nasar_match = nasar_pattern.search(line)
                if nasar_match:
                    train_nasar.append(float(nasar_match.group(1)))
            
            elif 'acc' in line:
                acc_match = acc_pattern.search(line)
                if acc_match:
                    val_accs.append(float(acc_match.group(1)))
            
            elif 'Dice score:' in line:
                dice_match = dice_pattern.search(line)
                if dice_match:
                    val_dice_scores.append(float(dice_match.group(1)))
    
    return train_losses, val_accs, val_dice_scores, train_nasar

def plot_metrics(train_losses, val_accs, val_dice_scores, train_nasar, folder="plots"):
    # Create output folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    epochs = range(1, len(train_losses) + 1)

    # ----------- Training Loss Plot -----------
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs[::5])  # Show every 5th tick
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{folder}/loss_{timestamp}.png")
    plt.show()
    plt.close()

    # ----------- Validation Accuracy Plot -----------
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.xticks(epochs[::5])
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{folder}/accuracy_{timestamp}.png")
    plt.show()        
    plt.close()

    # ----------- Dice Score Plot -----------
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_dice_scores, label='Dice Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.title('Dice Score')
    plt.xticks(epochs[::5])
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{folder}/dice_{timestamp}.png")
    plt.show()
    plt.close()

    # ----------- NASAR Plot ----------- 
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_nasar, label='Training NASAR', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('NASAR')
    plt.title('Network Average Spiking Activity Rate (NASAR)')
    plt.xticks(epochs[::5])
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{folder}/nasar_{timestamp}.png")
    plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    log_file_path = "extract.txt"
    train_losses, val_accs, val_dice_scores, train_nasar = extract_metrics_from_log(log_file_path)
    
    print("Extracted Metrics:")
    print("train_losses =", train_losses)
    print("val_accs =", val_accs)
    print("val_dice_scores =", val_dice_scores)
    print("train_nasar =", train_nasar)
    
    # Plot the metrics
    plot_metrics(train_losses, val_accs, val_dice_scores, train_nasar)