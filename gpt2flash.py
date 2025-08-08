import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load models
# Model A: Regular attention (eager)
model_regular = AutoModelForCausalLM.from_pretrained(
    "gpt2", 
    torch_dtype=torch.float16,
    attn_implementation="eager"  #Regular attention
).to(device).eval()

# Model B: FlashAttention2
model_flash = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  #Flash Attention 2
).to(device).eval()

prompt = "GPT2 is a model developed by OpenAI."
model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Number of tokens to generate for benchmarking
token_counts = [100] #list(range(10, 200, 1))  # 10, 11, 12, ..., 199

times_regular = []
times_flash = []

with torch.no_grad():
    for N in token_counts:
        # Regular attention timing
        start = time.time()
        model_regular.generate(**model_inputs, max_new_tokens=N, do_sample=True, temperature=0.9)
        times_regular.append(time.time() - start)

        # FlashAttention2 timing
        start = time.time()
        model_flash.generate(**model_inputs, max_new_tokens=N, do_sample=True, temperature=0.9)
        times_flash.append(time.time() - start)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(token_counts, times_regular, label="Regular Attention (GPU)", marker='o')
plt.plot(token_counts, times_flash, label="FlashAttention2 (GPU)", marker='o')
plt.xlabel("Number of generated tokens (N)")
plt.ylabel("Inference Time (seconds)")
plt.title("Inference Time vs. Generated Tokens")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('time_seq_gpu.png')