import time
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from transformers import DecisionTransformerConfig
from transformers import get_scheduler

from decision_transformer.models_flash.decision_transformer import AutonomousFlashFreeFlyerTransformer
import decision_transformer.manage as ART_manager
from decision_transformer.manage import device

import torch
from torch.optim import AdamW
from accelerate import Accelerator

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Settings
N = 350 
batch_size = 1
seq_length = 1
do_bechmarking = False


def benchmark_model(model, N, device):
    model.eval()
    torch.cuda.empty_cache()
    
    # Initialize dummy inputs
    states = torch.randn(batch_size, seq_length, model.state_dim).to(device)
    actions = torch.randn(batch_size, seq_length, model.act_dim).to(device)
    goal = torch.randn(batch_size, seq_length, model.state_dim).to(device)
    constraints_to_go = torch.randn(batch_size, seq_length, 1).to(device)
    returns_to_go = torch.randn(batch_size, seq_length, 1).to(device)
    timesteps = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long).to(device)

    start_time = time.time()
    with torch.no_grad():
        for i in range(N):
            state_preds, action_preds = model(
                states, actions, goal, returns_to_go, constraints_to_go, timesteps, attention_mask=attention_mask
            )
            # Get last prediction and append it to grow the sequence
            new_state = state_preds[:, -1:].detach()
            new_action = action_preds[:, -1:].detach()
            new_return = torch.randn(batch_size, 1, 1).to(device)  # Dummy return for the new state
            new_goal = torch.randn(batch_size, 1, model.state_dim).to(device)
            new_constraint = torch.randn(batch_size, 1, 1).to(device)
            
            # Append new predictions to the sequence
            states = torch.cat([states, new_state], dim=1)
            print("time step", i,"here shape", states.shape)
            actions = torch.cat([actions, new_action], dim=1)
            goal = torch.cat([goal, new_goal], dim=1)
            constraints_to_go = torch.cat([constraints_to_go, new_constraint], dim=1)
            returns_to_go = torch.cat([returns_to_go, new_return], dim=1)
            timesteps = torch.cat([timesteps, torch.randint(0, 1000, (batch_size, 1)).to(device)], dim=1)
            attention_mask = torch.ones(batch_size, states.size(1), dtype=torch.long).to(device)

    
    end_time = time.time()
    return end_time - start_time

# Initialize both models
model_flash = AutonomousFlashFreeFlyerTransformer(
    state_dim=6, act_dim=3, hidden_size=384, n_layer=6, n_head=8, n_positions=1000 * 10,
    max_ep_len=1000, use_flash_atten=True
).to(device)

model_no_flash = AutonomousFlashFreeFlyerTransformer(
    state_dim=6, act_dim=3, hidden_size=384, n_layer=6, n_head=8, n_positions=1000 * 10,
    max_ep_len=1000, use_flash_atten=False
).to(device)

time_flash = benchmark_model(model_flash, N, device)
time_no_flash = benchmark_model(model_no_flash, N, device)
print("FlashAttention Inference Time:", time_flash)
print("Regular Attention Inference Time:", time_no_flash)

if do_bechmarking:
    # Benchmarking against different N values
    N = np.arange(10, 400, 1)  # 10, 11, 12, ..., 399
    time_flash_withN =[]
    time_no_flash_withN = []
    for n in N:
        time_flash = benchmark_model(model_flash, n, device)
        time_no_flash = benchmark_model(model_no_flash, n, device)
        time_flash_withN.append(time_flash)
        time_no_flash_withN.append(time_no_flash)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(N, time_flash_withN, label="FlashAttention", marker='o', color='green')
    plt.plot(N, time_no_flash_withN, label="Regular Attention", marker='o', color='blue')
    plt.xlabel("Number of generated tokens (N), Batch Size = 1")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Time vs. Generated Tokens")
    plt.legend()
    plt.grid(True)
    plt.savefig('time_seqiwthN_gpu.png')

    # Bechmarking against differnt batch sizes with fixed N = 100
    batch_sizes = [1, 2, 4, 8, 16, 32]
    time_flash_withB = []
    time_no_flash_withB = []
    for batch_size in batch_sizes:
        time_flash = benchmark_model(model_flash, 100, device)
        time_no_flash = benchmark_model(model_no_flash, 100, device)
        time_flash_withB.append(time_flash)
        time_no_flash_withB.append(time_no_flash)
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, time_flash_withB, label="FlashAttention", marker='o', color='green')
    plt.plot(batch_sizes, time_no_flash_withB, label="Regular Attention", marker='o', color='blue')
    plt.xlabel("Batch Size")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Time vs. Batch Size, N = 100")
    plt.legend()
    plt.grid(True)
    plt.savefig('time_batch_gpu.png')
