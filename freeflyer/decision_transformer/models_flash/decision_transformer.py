import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D

import transformers

from .model import TrajectoryModel
from .trajectory_gpt2 import GPT2Model
from typing import Optional, Tuple, Union

class AutonomousFlashFreeFlyerTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            n_positions = 1024,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            use_flash_atten=True,
            activation_function = 'relu',
            **kwargs            
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_positions = n_positions,
            activation_function=activation_function,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(use_flash_atten, config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_constraint = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_goal = torch.nn.Linear(self.state_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        # self.predict_return = torch.nn.Linear(hidden_size, 1)  # we dont need this

    def forward(
        self,
        states,
        actions,
        goal,
        returns_to_go,
        constraints_to_go,
        timesteps,
        attention_mask=None,
        return_dict = None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goal)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        constraints_embeddings = constraints_embeddings + time_embeddings

        # this makes the sequence look like (g_1, R_1, C_1, s_1, a_1, g_2, R_2, C_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

       # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 5 * seq_length)
        )

        device = stacked_inputs.device
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
        )

        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:, 4])  # predict next return given state and action
        state_preds = self.predict_state(x[:, 4])  # predict next state given (T+R+C)+state and action
        action_preds = self.predict_action(x[:, 3])  # predict next action given (T+R+C)+state
    
        return state_preds, action_preds
    

'''
The model below is stochastic transformer model for trajectory prediction. It will
learn the mean and variance of the action distribtion. 
'''

class AutonomousStochasticFlashFreeFlyerTransformer(TrajectoryModel):

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            n_positions,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            min_std=1e-4,
            max_std=1.0,
            use_flash_atten=True,
            activation_function = 'relu',
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.action_tanh = action_tanh
        self.min_std = min_std
        self.max_std = max_std

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter
            n_embd=hidden_size,
            n_positions = n_positions,
            activation_function = activation_function,
            **kwargs
        )

        self.transformer = GPT2Model(use_flash_atten, config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_constraint = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_goal = torch.nn.Linear(self.state_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)

        # For stochastic action output: mean + log_std
        self.action_mean = nn.Linear(hidden_size, self.act_dim)
        self.action_log_std_layer = nn.Linear(hidden_size, self.act_dim) # Dynamic std
        # self.action_log_std = nn.Parameter(torch.zeros(self.act_dim))  # Static std

    def forward(
        self,
        states,
        actions,
        goal,
        returns_to_go,
        constraints_to_go,
        timesteps,
        attention_mask=None,
        return_dict=False):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embeddings
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        goal_embeddings = self.embed_goal(goal)
        returns_embeddings = self.embed_return(returns_to_go)
        constraints_embeddings = self.embed_constraint(constraints_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Time embedding added
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        goal_embeddings += time_embeddings
        returns_embeddings += time_embeddings
        constraints_embeddings += time_embeddings

        # Stack sequence as (g, R, C, s, a, g, R, C, s, a, ...)
        stacked_inputs = torch.stack(
            (goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings),
            dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Stack attention mask
        stacked_attention_mask = (
            torch.stack([attention_mask]*5, dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 5 * seq_length)
        )

        device = stacked_inputs.device

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
        )

        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = self.predict_state(x[:, 4])  # predict next state from a_t
        action_mean = self.action_mean(x[:, 3])    # predict action mean from s_t

        ######## static std dev
        # action_std = torch.clamp(self.action_log_std.exp(), self.min_std, self.max_std) 
        ######## dyanmic std
        action_log_std = self.action_log_std_layer(x[:, 3])  # predict log_std from hidden state
        action_std = torch.clamp(action_log_std.exp(), self.min_std, self.max_std)

        return state_preds, action_mean, action_std

    def get_action_distribution(self, action_mean, action_std):
        if self.action_tanh:
            # Use TanhNormal distribution for bounded actions
            base_dist = D.Normal(action_mean, action_std)
            action_dist = D.TransformedDistribution(base_dist, [D.transforms.TanhTransform(cache_size=1)])
        else:
            action_dist = D.Normal(action_mean, action_std)
        return action_dist
