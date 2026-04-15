from llama_cpp import Llama
import numpy as np

llm = Llama(model_path='models/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf', 
            n_ctx=512, verbose=False, n_gpu_layers=0, logits_all=True)

prompt = '<|begin_of_text|><|header_start|>user<|header_end|>\n\nWhat is 2+2?<|eot|><|header_start|>assistant<|header_end|>\n\n'
tokens = llm.tokenize(prompt.encode(), add_bos=False, special=True)
print(f'Input tokens ({len(tokens)}): {tokens}')

# Generate token by token
llm.eval(tokens)
for step in range(20):
    logits_np = np.array(llm.scores[llm.n_tokens - 1])
    top5_ids = np.argsort(logits_np)[-5:][::-1]
    top5_vals = logits_np[top5_ids]
    tok_id = int(top5_ids[0])
    tok_str = llm.detokenize([tok_id]).decode('utf-8', errors='replace')
    pairs = [(int(i), f"{v:.4f}") for i, v in zip(top5_ids, top5_vals)]
    print(f'Step {step}: token={tok_id} ({repr(tok_str)}) top5={pairs}')
    if tok_id == llm.token_eos():
        break
    llm.eval([tok_id])
