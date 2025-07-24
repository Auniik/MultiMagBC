import os
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def save_attention_heatmap(attn_dict, save_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(attn_dict['weights'], annot=True, fmt=".2f", cmap='viridis',
                xticklabels=attn_dict['magnifications'],
                yticklabels=attn_dict['magnifications'])
    plt.title("Hierarchical Attention (Mean)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()