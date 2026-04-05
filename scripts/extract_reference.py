from llama_cpp import Llama
import numpy as np
import ctypes

# Load model with logits
llm = Llama(model_path='models/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf', 
            n_ctx=512, verbose=False, n_gpu_layers=0, logits_all=True)

# Check model metadata
print("=== Model Info ===")
print(f"n_vocab: {llm.n_vocab()}")
print(f"n_ctx_train: {llm.n_ctx()}")
print(f"n_embd: {llm.n_embd()}")

# Tokenize our test prompt
prompt = '<|begin_of_text|><|header_start|>user<|header_end|>\n\nWhat is 2+2?<|eot|><|header_start|>assistant<|header_end|>\n\n'
tokens = llm.tokenize(prompt.encode(), add_bos=False, special=True)
print(f"\n=== Tokens ({len(tokens)}) ===")
print(tokens)

# Also tokenize individual pieces to understand
pieces = ['<|begin_of_text|>', '<|header_start|>', 'user', '<|header_end|>', '\n\n', 
          'What', ' is', ' ', '2', '+', '2', '?', 
          '<|eot|>', '<|header_start|>', 'assistant', '<|header_end|>', '\n\n']
for p in pieces:
    try:
        t = llm.tokenize(p.encode(), add_bos=False, special=True)
        print(f"  '{p}' -> {t}")
    except:
        print(f"  '{p}' -> ERROR")

# Now generate and get per-step logits
print(f"\n=== Step-by-step generation (greedy) ===")
llm.eval(tokens)
generated = []
for step in range(10):
    logits_np = np.array(llm.scores[llm.n_tokens - 1])
    top10_ids = np.argsort(logits_np)[-10:][::-1]
    top10_vals = logits_np[top10_ids]
    tok_id = int(top10_ids[0])
    tok_str = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
    generated.append(tok_id)
    
    print(f"Step {step}: token={tok_id} ({repr(tok_str)})")
    for i, (tid, tv) in enumerate(zip(top10_ids, top10_vals)):
        ts = llm.detokenize([int(tid)]).decode('utf-8', errors='replace')
        print(f"  #{i+1}: [{tid}]={tv:.6f} ({repr(ts)})")
    
    if tok_id == llm.token_eos():
        break
    llm.eval([tok_id])

print(f"\nGenerated tokens: {generated}")

# Get embeddings for first few tokens to compare
print(f"\n=== Token embeddings (first 8 values) ===")
# llama-cpp doesn't easily expose embeddings, skip this

# Also test with "capital of France" which works in both modes
print(f"\n=== Second prompt: 'What is the capital of France?' ===")
llm.reset()
prompt2 = '<|begin_of_text|><|header_start|>user<|header_end|>\n\nWhat is the capital of France?<|eot|><|header_start|>assistant<|header_end|>\n\n'
tokens2 = llm.tokenize(prompt2.encode(), add_bos=False, special=True)
print(f"Tokens ({len(tokens2)}): {tokens2}")
llm.eval(tokens2)
for step in range(10):
    logits_np = np.array(llm.scores[llm.n_tokens - 1])
    top5_ids = np.argsort(logits_np)[-5:][::-1]
    top5_vals = logits_np[top5_ids]
    tok_id = int(top5_ids[0])
    tok_str = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
    print(f"Step {step}: token={tok_id} ({repr(tok_str)}) top=[{top5_ids[0]}]={top5_vals[0]:.4f}")
    if tok_id == llm.token_eos():
        break
    llm.eval([tok_id])
