import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from transformers import DecisionTransformerConfig
from transformers import get_scheduler

from decision_transformer.art import AutonomousStochasticFreeflyerTransformer
import decision_transformer.managestoch as ART_manager
from decision_transformer.managestoch import device

import torch
from torch.optim import AdamW
from accelerate import Accelerator
import time
import wandb

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initial parameters
model_name_4_saving = 'checkpoint_ff_art_stoch'
save_path = '/decision_transformer/saved_files/checkpoints/'
mdp_constr = True
datasets, dataloaders = ART_manager.get_train_val_test_data(mdp_constr=mdp_constr, timestep_norm=False)
train_loader, eval_loader, test_loader = dataloaders
n_state = train_loader.dataset.n_state
n_data = train_loader.dataset.n_data
n_action = train_loader.dataset.n_action
n_time = train_loader.dataset.max_len

# Transformer parameters
config = DecisionTransformerConfig(
    state_dim=n_state, 
    act_dim=n_action,
    hidden_size=384,
    max_ep_len=n_time,
    vocab_size=1,
    action_tanh=False,
    n_positions=1024,
    n_layer=6,
    n_head=6,
    n_inner=None,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    )

model = AutonomousStochasticFreeflyerTransformer(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")
model.to(device);

optimizer = AdamW(model.parameters(), lr=3e-5)
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=8)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_loader, eval_loader
)
num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
max_steps = 200000  # Stop training after N steps

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=10,
    num_training_steps=max_steps,
)

# Initialize wandb
wandb.init(
    project="autonomous_stochastic_Normal_DT_training",
    name="run_stoch_1",
    config={
        "learning_rate": 3e-5,
        "batch_size": 32,
        "epochs": 1,
        "gradient_accumulation_steps": 8,
    }
)

# Loss function
mse_loss = torch.nn.MSELoss()

print('\n======================')
print('Initializing training\n')
start_time_train = time.time()

# Training loop
num_epochs = 1
eval_every = 100
eval_iters = 100
save_every = 10
global_step = 0


for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader, start=0):
        if global_step >= max_steps:
            print(f"Reached max_steps {max_steps}. Stopping training.")
            break

        states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = batch

        # Forward pass — now outputs (state_preds, action_mean, action_std)
        state_preds, action_mean, action_std = model(
            states=states_i,
            actions=actions_i,
            goal=goal_i,
            returns_to_go=rtgs_i,
            constraints_to_go=ctgs_i,
            timesteps=timesteps_i,
            attention_mask=attention_mask_i,
            return_dict=False,
        )

        # Create action distribution tanh
        dist = model.get_action_distribution(action_mean, action_std)
        log_prob = dist.log_prob(actions_i)  # shape: (batch_size, seq_len, act_dim)

        # Losses
        action_loss = -log_prob.mean()  # NLL loss
        state_loss = mse_loss(state_preds[:, :-1, :], states_i[:, 1:, :])  # next state prediction MSE
        total_loss = state_loss + action_loss

        # Backward + optimize
        accelerator.backward(total_loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Log to wandb
        wandb.log({
            "train/state_loss": state_loss.item(),
            "train/action_loss": action_loss.item(),
            "train/total_loss": total_loss.item(),
            "train/learning_rate": lr_scheduler.get_last_lr()[0],
            "global_step": global_step
        }, step=global_step)

        # Eval loop
        if global_step % eval_every == 0 and global_step != 0:
            model.eval()
            losses, losses_state, losses_action = [], [], []

            with torch.no_grad():
                for _ in range(eval_iters):
                    data_iter = iter(eval_dataloader)
                    states_i, actions_i, rtgs_i, ctgs_i, goal_i, timesteps_i, attention_mask_i, _, _, _ = next(data_iter)

                    # Move to device
                    states_i, actions_i = states_i.to(device), actions_i.to(device)
                    rtgs_i, ctgs_i, goal_i = rtgs_i.to(device), ctgs_i.to(device), goal_i.to(device)
                    timesteps_i, attention_mask_i = timesteps_i.to(device), attention_mask_i.to(device)

                    # Forward pass
                    state_preds, action_mean, action_std = model(
                        states=states_i,
                        actions=actions_i,
                        goal=goal_i,
                        returns_to_go=rtgs_i,
                        constraints_to_go=ctgs_i,
                        timesteps=timesteps_i,
                        attention_mask=attention_mask_i,
                        return_dict=False,
                    )

                    dist = model.get_action_distribution(action_mean, action_std)
                    log_prob = dist.log_prob(actions_i)
                    action_loss_i = -log_prob.mean()
                    state_loss_i = mse_loss(state_preds[:, :-1, :], states_i[:, 1:, :])

                    losses.append(accelerator.gather(state_loss_i + action_loss_i))
                    losses_state.append(accelerator.gather(state_loss_i))
                    losses_action.append(accelerator.gather(action_loss_i))

            loss = torch.mean(torch.tensor(losses))
            loss_state = torch.mean(torch.tensor(losses_state))
            loss_action = torch.mean(torch.tensor(losses_action))
            print(f"Step {global_step}: Eval loss: {loss.item()}, State loss: {loss_state.item()}, Action loss: {loss_action.item()}")

            wandb.log({
                "eval/loss": loss.item(),
                "eval/state_loss": loss_state.item(),
                "eval/action_loss": loss_action.item(),
                "global_step": global_step
            }, step=global_step)
            model.train()

        if global_step % save_every == 0 and global_step != 0:
            print(f"Step {global_step}: Saving model checkpoint...")
            accelerator.save_state(root_folder + save_path + model_name_4_saving)

        global_step += 1