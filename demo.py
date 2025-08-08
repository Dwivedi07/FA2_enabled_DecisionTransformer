from transformers import GPT2Config, DecisionTransformerConfig, DecisionTransformerGPT2Model
from transformers.models.decision_transformer.configuration_decision_transformer import DecisionTransformerConfig
from transformers.models.decision_transformer.modeling_decision_transformer import DecisionTransformerPreTrainedModel, DecisionTransformerGPT2Model, DecisionTransformerOutput

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast

class AutonomousFreeflyerTransformer(DecisionTransformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        # Manually modify the GPT-2 config to include FlashAttention
        gpt2_config = GPT2Config.from_pretrained('gpt2', attn_implementation='flash_attention_2')  # Enable FlashAttention
        # You can copy values from your DecisionTransformerConfig to gpt2_config if necessary, e.g. hidden_size, n_layer, etc.
        gpt2_config.n_layer = config.n_layer
        gpt2_config.n_head = config.n_head
        gpt2_config.hidden_size = config.hidden_size
        gpt2_config.n_positions = config.n_positions
        
        # Initialize the GPT2 model with the FlashAttention configuration
        self.encoder = DecisionTransformerGPT2Model(config=gpt2_config)

        # The rest of the initialization remains the same
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_goal = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_constraint = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )

        # Initialize weights and apply final processing
        self.post_init()

# Transformer parameters
config = DecisionTransformerConfig(
    state_dim=6, 
    act_dim=3,
    hidden_size=384,
    max_ep_len=100,
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


device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutonomousFreeflyerTransformer(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT size: {model_size/1000**2:.1f}M parameters")
model.to(device);


####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################



# from transformers import GPT2Config, DecisionTransformerConfig, GPT2Model
# import torch
# import torch.nn as nn
# from torch.cuda.amp import autocast

# class AutonomousFreeflyerTransformer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size

#         # Modify the GPT-2 config to include FlashAttention
#         gpt2_config = GPT2Config.from_pretrained('gpt2', attn_implementation='flash_attention_2')
#         gpt2_config.n_layer = config.n_layer
#         gpt2_config.n_head = config.n_head
#         gpt2_config.hidden_size = config.hidden_size
#         gpt2_config.n_positions = config.n_positions


#         # Initialize the GPT2 model with the FlashAttention configuration
#         self.encoder = GPT2Model(config=gpt2_config)

#         # Rest of the initialization remains the same
#         self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
#         self.embed_goal = torch.nn.Linear(config.state_dim, config.hidden_size)
#         self.embed_return = torch.nn.Linear(1, config.hidden_size)
#         self.embed_constraint = torch.nn.Linear(1, config.hidden_size)
#         self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
#         self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

#         self.embed_ln = nn.LayerNorm(config.hidden_size)

#         self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
#         self.predict_action = nn.Sequential(
#             *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
#         )

    

#     def forward(self, states, actions, goal, returns_to_go, constraints_to_go, timesteps, attention_mask=None):
#         batch_size, seq_length = states.shape[0], states.shape[1]

#         # Default attention mask
#         if attention_mask is None:
#             attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

#         # Embed each modality with a different head
#         state_embeddings = self.embed_state(states)
#         action_embeddings = self.embed_action(actions)
#         goal_embeddings = self.embed_goal(goal)
#         returns_embeddings = self.embed_return(returns_to_go)
#         constraints_embeddings = self.embed_constraint(constraints_to_go)
#         time_embeddings = self.embed_timestep(timesteps)

#         # Time embeddings are treated similarly to positional embeddings
#         state_embeddings += time_embeddings
#         action_embeddings += time_embeddings
#         goal_embeddings += time_embeddings
#         returns_embeddings += time_embeddings
#         constraints_embeddings += time_embeddings

#         # Stack inputs
#         stacked_inputs = torch.stack((goal_embeddings, returns_embeddings, constraints_embeddings, state_embeddings, action_embeddings), dim=1)
#         stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, 5 * seq_length, self.hidden_size)
#         stacked_inputs = self.embed_ln(stacked_inputs)

#         # Stack attention mask
#         stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1)
#         stacked_attention_mask = stacked_attention_mask.permute(0, 2, 1).reshape(batch_size, 5 * seq_length)

#         # Use Automatic Mixed Precision (AMP) during the forward pass
#         with autocast(device_type="cuda", dtype=torch.float16):  # Automatic mixed precision for float16
#             encoder_outputs = self.encoder(
#                 inputs_embeds=stacked_inputs,
#                 attention_mask=stacked_attention_mask
#             )
            
#             x = encoder_outputs.last_hidden_state

#         # Reshape the output
#         x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

#         # Predict state and action
#         state_preds = self.predict_state(x[:, 4])  # Predict next state from the action
#         action_preds = self.predict_action(x[:, 3])  # Predict next action from the state

#         return state_preds, action_preds


# # Initialize the model with DecisionTransformerConfig
# config = DecisionTransformerConfig(
#     state_dim=6, 
#     act_dim=3,
#     hidden_size=384,
#     max_ep_len=100,
#     vocab_size=1,
#     action_tanh=False,
#     n_positions=1024,
#     n_layer=6,
#     n_head=6,
#     n_inner=None,
#     resid_pdrop=0.1,
#     embd_pdrop=0.1,
#     attn_pdrop=0.1,
# )

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Initialize the model and send it to the appropriate device
# model = AutonomousFreeflyerTransformer(config)
# model_size = sum(t.numel() for t in model.parameters())
# print(f"GPT size: {model_size/1000**2:.1f}M parameters")
# model.to(device)

