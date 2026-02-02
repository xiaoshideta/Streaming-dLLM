import torch

def deterministic_window_tail_sampler(tokens, window=32, tail_keep=3):
    L = len(tokens)
    if L == 0:
        return torch.tensor([], device=tokens.device, dtype=torch.long)
    
    head_tokens = tokens[:min(int(window), L)]

    if L <= window:
        return head_tokens

    tail_keep = min(tail_keep, max(L - len(head_tokens), 0))
    tail_tokens = tokens[-tail_keep:] if tail_keep > 0 else torch.tensor([], device=tokens.device, dtype=torch.long)


    final_tokens = torch.cat([head_tokens, tail_tokens]).long()

    return final_tokens
    
