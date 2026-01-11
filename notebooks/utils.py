from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch, os, json, random, cv2, textwrap


def show_samples(df, img_dir):
    """
    Display images in one row with Question & Answer shown BELOW each image.
    df must contain: img_name, question, answer
    """
    n = len(df)
    plt.figure(figsize=(4*n, 7))  # height increased for Q&A text

    for i, row in enumerate(df.itertuples(), 1):
        img_path = os.path.join(img_dir, row.img_name)

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create subplot
        ax = plt.subplot(1, n, i)
        ax.imshow(img)
        ax.axis("off")

        plt.title(
            f"Sample {i}", 
            fontsize=14,
            loc='center'
        )
        
        # Wrap text nicely (adjust width depending on image size)
        q_text = textwrap.fill(f"Q: {row.question}", width=40)
        a_text = textwrap.fill(f"A: {row.answer}", width=40)

        # Add text BELOW the image using .text()
        ax.text(
            0.5, -0.15, f"{q_text}\n{a_text}",
            ha="center", va="top",
            fontsize=12,
            transform=ax.transAxes
        )

    plt.tight_layout()
    plt.show()


def plot_graph(train_losses, val_losses, train_accs, val_accs, num_epochs=50):
    epochs_range = range(1, num_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs_range, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_samples(
    results,
    dataset_dir,
    num_samples=5,
    only_correct=None,   # None = all, True = correct only, False = incorrect only
    shuffle=True,
    figsize=(15, 5)
):
    # Optional filtering
    if only_correct is not None:
        results = [r for r in results if r["correct"] == only_correct]

    if len(results) == 0:
        print("No samples to display after filtering.")
        return

    # Sampling
    if shuffle:
        samples = random.sample(results, min(num_samples, len(results)))
    else:
        samples = results[:num_samples]

    # Plot
    fig, axes = plt.subplots(1, len(samples), figsize=figsize)

    if len(samples) == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        image_path = os.path.join(dataset_dir, 'Slake1.0', 'imgs', sample["image_path"])
        img = Image.open(image_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        title = (
            f"Q: {sample['question']}\n"
            f"GT: {sample['true_answer']}\n"
            f"Pred: {sample['predicted_answer']}\n"
            f"Conf: {sample['confidence']:.2f}\n"
            f"{'✅ Correct' if sample['correct'] else '❌ Wrong'}"
        )

        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()
