from __future__ import annotations
import torch 
def top_k_top_p_filtering(logits:torch.Tensor,top_K:int|None=None,top_P:float|None=None):
    """
    Filter a distribution of logits using top=k and/or nuceleus(top_p) filtering
    -logits:(B,vocab)
    Returns filtered logits with -inf for masked entries

    """
    B,V=logits.shape
    filtered=logits.clone()

    if top_K is not None and top_K<V:
        topk_vals=torch.topk(filtered,top_K,dim=-1)
        kth=topk_vals[:,-1].unsqueeze(-1)
        filtered[filtered<kth]=float('-inf')

    if top_P is not None and 0<top_P<1.0:
        sorted_logits,sorted_idx=torch.sort(filtered,descending=True,dim=-1)
        probs=torch.softmax(sorted_logits,dim=-1)
        cumsum=torch.cumsum(probs,dim=-1)
        mask=cumsum>top_P
        # keep at least 1 token
        mask[...,0]=False
        sorted_logits[mask]=float('-inf')
        # scatter back
        filtered=torch.full_like(filtered,float('-inf'))
        filtered.scatter_(1,sorted_idx,sorted_logits)

    return filtered
