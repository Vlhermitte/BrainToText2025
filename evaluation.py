import numpy as np
from typing import List

import torch
from torch import nn

from data import ascii_ids_to_text


@torch.no_grad()
def greedy_decode(logits: torch.Tensor, blank_id=128) -> List[str]:
    """
        logits: (T, B, C) raw output
        Returns list of lists of predicted token IDs (per batch)
        """
    best = logits.argmax(dim=-1)  # (T, B)
    best = best.cpu().numpy()
    results = []
    for b in range(best.shape[1]):
        seq = []
        prev = blank_id
        for t in range(best.shape[0]):
            p = best[t, b]
            if p != blank_id and p != prev:
                seq.append(p)
            prev = p
        results.append(seq)
    return results


def predict_sentence(model: nn.Module, x: torch.Tensor, blank_id=128) -> str:
    """
    x: (T, 512) input features for one sample
    returns predicted text and list of predicted IDs
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = x.unsqueeze(1).to(device)  # (T, 1, 512)
        x_len = torch.tensor([x.shape[0]], dtype=torch.long).to(device)  # (1,)
        logits, _ = model(x, x_len)  # (T, 1, C)

        decoded_ids = greedy_decode(logits, blank_id=blank_id)[0]
        decoded_text = ascii_ids_to_text(decoded_ids)

    return decoded_text, logits.argmax(dim=-1).squeeze(1).cpu().tolist()


def run_evaluate(model, dataloader, vocab_size=129, blank_id=128, device='cpu'):
    model.eval()
    cer_total, wer_total = 0.0, 0.0
    n = 0

    with torch.no_grad():
        for x_pad, x_len, targets, target_len in dataloader:
            # move to device
            x_pad = x_pad.to(device)
            x_len = x_len.to(device)
            # transpose for (T,B,F)
            x_tbF = x_pad.permute(1,0,2).contiguous()

            # forward
            logits, in_len = model(x_tbF, x_len)
            log_probs = logits.log_softmax(dim=-1)

            # greedy decode
            pred_ids_batch = greedy_decode(log_probs, blank_id=blank_id)

            # gather reference strings (you may store them in dataset.df)
            batch_refs = []
            for i in range(len(x_len)):
                start = sum(target_len[:i])
                end = start + target_len[i]
                ref_ids = targets[start:end].cpu().tolist()
                batch_refs.append(ascii_ids_to_text(ref_ids))

            # convert predicted ids → text
            batch_hyps = [ascii_ids_to_text(ids) for ids in pred_ids_batch]

            # compute metrics
            for ref, hyp in zip(batch_refs, batch_hyps):
                cer_total += character_error_rate(ref, hyp)
                wer_total += word_error_rate(ref, hyp)
                n += 1

    avg_cer = cer_total / n
    avg_wer = wer_total / n
    print(f"Test Character Error Rate (Levenshtein distance normalized by reference length): {avg_cer:.3f}")
    print(f"Word Error Rate (Levenshtein distance on words): {avg_wer:.3f}")
    return avg_cer, avg_wer

def levenshtein(a, b):
    dp = np.zeros((len(a)+1, len(b)+1), dtype=np.int32)
    dp[:,0] = np.arange(len(a)+1)
    dp[0,:] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    return dp[len(a), len(b)]

def character_error_rate(ref: str, hyp: str) -> float:
    return levenshtein(ref, hyp) / max(1, len(ref))

def word_error_rate(ref: str, hyp: str) -> float:
    ref_w, hyp_w = ref.split(), hyp.split()
    return levenshtein(ref_w, hyp_w) / max(1, len(ref_w))