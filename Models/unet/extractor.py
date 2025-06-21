import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set seaborn style and color palette
sns.set_style("darkgrid")
sns.set_palette("husl")

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

def plot_metrics_comparison(file1_data, file2_data, file1_name, file2_name, folder="plots"):
    """
    Plot metrics from two files in 4 subplots for comparison
    Handles cases where some metrics might be missing from one or both files
    """
    # Create output folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Unpack data from both files
    train_losses1, val_accs1, val_dice_scores1, train_nasar1 = file1_data
    train_losses2, val_accs2, val_dice_scores2, train_nasar2 = file2_data
    
    # Create epochs range for each file
    epochs1 = range(1, len(train_losses1) + 1) if train_losses1 else []
    epochs2 = range(1, len(train_losses2) + 1) if train_losses2 else []
    
    # Create 2x2 subplot figure with seaborn styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define colors using seaborn palette
    colors = sns.color_palette("husl", 2)
    color1, color2 = colors[0], colors[1]
    
    # ----------- Training Loss Plot -----------
    if train_losses1:
        ax1.plot(epochs1, train_losses1, label=f'{file1_name}', color=color1, linewidth=2.5, marker='o', markersize=3)
    if train_losses2:
        ax1.plot(epochs2, train_losses2, label=f'{file2_name}', color=color2, linewidth=2.5, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    if train_losses1 or train_losses2:
        ax1.legend(fontsize=11)
    else:
        ax1.text(0.5, 0.5, 'No Training Loss Data', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    
    # ----------- Validation Accuracy Plot -----------
    if val_accs1:
        epochs1_acc = range(1, len(val_accs1) + 1)
        ax2.plot(epochs1_acc, val_accs1, label=f'{file1_name}', color=color1, linewidth=2.5, marker='o', markersize=3)
    if val_accs2:
        epochs2_acc = range(1, len(val_accs2) + 1)
        ax2.plot(epochs2_acc, val_accs2, label=f'{file2_name}', color=color2, linewidth=2.5, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    if val_accs1 or val_accs2:
        ax2.legend(fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'No Validation Accuracy Data', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    # ----------- Dice Score Plot -----------
    if val_dice_scores1:
        epochs1_dice = range(1, len(val_dice_scores1) + 1)
        ax3.plot(epochs1_dice, val_dice_scores1, label=f'{file1_name}', color=color1, linewidth=2.5, marker='o', markersize=3)
    if val_dice_scores2:
        epochs2_dice = range(1, len(val_dice_scores2) + 1)
        ax3.plot(epochs2_dice, val_dice_scores2, label=f'{file2_name}', color=color2, linewidth=2.5, marker='s', markersize=3)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Dice Score', fontsize=12)
    ax3.set_title('Dice Score Comparison', fontsize=14, fontweight='bold')
    if val_dice_scores1 or val_dice_scores2:
        ax3.legend(fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'No Dice Score Data', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # ----------- NASAR Plot -----------
    if train_nasar1:
        epochs1_nasar = range(1, len(train_nasar1) + 1)
        ax4.plot(epochs1_nasar, train_nasar1, label=f'{file1_name}', color=color1, linewidth=2.5, marker='o', markersize=3)
    if train_nasar2:
        epochs2_nasar = range(1, len(train_nasar2) + 1)
        ax4.plot(epochs2_nasar, train_nasar2, label=f'{file2_name}', color=color2, linewidth=2.5, marker='s', markersize=3)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('NASAR', fontsize=12)
    ax4.set_title('Network Average Spiking Activity Rate (NASAR) Comparison', fontsize=14, fontweight='bold')
    if train_nasar1 or train_nasar2:
        ax4.legend(fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'No NASAR Data', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # Adjust layout and save in multiple formats
    plt.tight_layout()
    
    # Save as EPS for LaTeX
    plt.savefig(f"{folder}/metrics_comparison_{timestamp}.eps", format='eps', dpi=300, bbox_inches='tight')
    
    # Save as PDF for LaTeX
    plt.savefig(f"{folder}/metrics_comparison_{timestamp}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    
    # Also save as PNG for quick viewing
    plt.savefig(f"{folder}/metrics_comparison_{timestamp}.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

def plot_metrics(train_losses, val_accs, val_dice_scores, train_nasar, folder="plots"):
    # Create output folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    epochs = range(1, len(train_losses) + 1)

    # Set seaborn style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = sns.color_palette("husl", 4)

    # ----------- Training Loss Plot -----------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color=colors[0], linewidth=2.5, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xticks(epochs[::5])  # Show every 5th tick
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{folder}/loss_{timestamp}.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/loss_{timestamp}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/loss_{timestamp}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # ----------- Validation Accuracy Plot -----------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accs, label='Val Accuracy', color=colors[1], linewidth=2.5, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(epochs[::5])
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{folder}/accuracy_{timestamp}.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/accuracy_{timestamp}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/accuracy_{timestamp}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()        
    plt.close()

    # ----------- Dice Score Plot -----------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_dice_scores, label='Dice Score', color=colors[2], linewidth=2.5, marker='^', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice', fontsize=12)
    plt.title('Dice Score', fontsize=14, fontweight='bold')
    plt.xticks(epochs[::5])
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{folder}/dice_{timestamp}.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/dice_{timestamp}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/dice_{timestamp}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # ----------- NASAR Plot ----------- 
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_nasar, label='Training NASAR', color=colors[3], linewidth=2.5, marker='d', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('NASAR', fontsize=12)
    plt.title('Network Average Spiking Activity Rate (NASAR)', fontsize=14, fontweight='bold')
    plt.xticks(epochs[::5])
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{folder}/nasar_{timestamp}.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/nasar_{timestamp}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f"{folder}/nasar_{timestamp}.png", format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    # # For single file analysis
    # log_file_path = "extract.txt"
    # train_losses, val_accs, val_dice_scores, train_nasar = extract_metrics_from_log(log_file_path)
    
    # print("Extracted Metrics:")
    # print("train_losses =", train_losses)
    # print("val_accs =", val_accs)
    # print("val_dice_scores =", val_dice_scores)
    # print("train_nasar =", train_nasar)
    
    # # Plot the metrics for single file
    # plot_metrics(train_losses, val_accs, val_dice_scores, train_nasar)

    #_________________________________________________________________#

    # For comparing two files - uncomment and modify as needed
    # Example for comparing two log files
    file1_path = "U-Net.txt"
    file2_path = "CSA-SNN.txt"
    
    # Extract metrics from both files
    file1_data = extract_metrics_from_log(file1_path)
    file2_data = extract_metrics_from_log(file2_path)
    
    # Get filenames without extension for labels
    file1_name = os.path.splitext(os.path.basename(file1_path))[0]
    file2_name = os.path.splitext(os.path.basename(file2_path))[0]
    
    # Create comparison plot
    plot_metrics_comparison(file1_data, file2_data, file1_name, file2_name)
