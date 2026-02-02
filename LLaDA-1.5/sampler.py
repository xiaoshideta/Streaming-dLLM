import torch
import torch.nn.functional as F

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



def exponential_importance_sampler(tokens, total_budget=None, keep_ratio=0.1, alpha=50.0, beta=10.0):
    L = len(tokens)
    if L == 0:
        return torch.tensor([], device=tokens.device, dtype=torch.long)
    if L <= 32:
        return tokens
    else:
        if total_budget is None:
            total_budget = max(1, int(L * keep_ratio))
        
        if total_budget >= L:
            return tokens
        positions = torch.arange(L, device=tokens.device, dtype=torch.float32)
        head_score = torch.exp(-positions / alpha)
        tail_score = torch.exp(-(L - 1 - positions) / beta)
        scores = head_score + tail_score 
        _, selected_indices = torch.topk(scores, total_budget, sorted=False)
        selected_indices = selected_indices.sort().values
    
    return tokens[selected_indices]

def gaussian_importance_sampler(tokens, keep_num, sigma_head=30.0, sigma_tail=10.0):
    L = len(tokens)
    if L == 0: return torch.tensor([], device=tokens.device)
    if keep_num >= L: return tokens

    x = torch.arange(L, device=tokens.device, dtype=torch.float32)
    gauss_head = torch.exp(-(x**2) / (2 * sigma_head**2))

    gauss_tail = torch.exp(-((x - (L-1))**2) / (2 * sigma_tail**2))

    importance = gauss_head + gauss_tail
    
    _, indices = torch.topk(importance, keep_num, sorted=False)
    return tokens[indices.sort().values]
