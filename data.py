from typing import List, Union, Tuple

import pandas as pd
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

class NeuralDataset(Dataset):
    """
    Expects a DataFrame with columns:
      - 'neural_features': np.ndarray shape (T, 512)  (dtype float or similar)
      - 'transcriptions': str (ASCII) or List[int] in 0..127
      - 'sentence_label': str (used to determine target length)
    """
    def __init__(self, df: pd.DataFrame, blank_id: int = 127):
        self.df = df.reset_index(drop=True)
        self.blank_id = blank_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        x_arr = self.df.iloc[idx]['neural_features']   # (T, 512)
        x = torch.tensor(x_arr, dtype=torch.float32)   # (T, 512)

        y_field: Union[str, List[int], np.ndarray] = self.df.iloc[idx]['transcriptions']
        if isinstance(y_field, str):
            y = text_to_ascii_ids(y_field)             # (U,)
        else:
            y = torch.tensor(y_field, dtype=torch.long)

        # sanity: ensure targets have no BLANK_ID
        if (y == self.blank_id).any():
            raise ValueError("Targets must not contain BLANK_ID.")

        # remove padding from y. Take only the first len(self.df.iloc[idx]['sentence_label']) elements
        len_sentence = len(self.df.iloc[idx]['sentence_label'])
        y = y[:len_sentence]

        return x, y

def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    # Unpack
    xs, ys = zip(*batch)  # xs: list of (T_i, 512); ys: list of (U_i,)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    # Pad features to max_T
    max_T = int(max(lengths))
    feat_dim = xs[0].shape[1]
    padded = torch.zeros(max_T, len(xs), feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        padded[:T, i, :] = x

    # Concatenate targets
    target_lengths = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    targets = torch.cat(ys, dim=0) if len(ys) > 1 else ys[0]

    return padded, lengths, targets, target_lengths

def text_to_ascii_ids(s: str) -> torch.Tensor:
    # Strip to ASCII, ignore others
    b = s.encode("ascii", "ignore")
    return torch.tensor(list(b), dtype=torch.long)

def ascii_ids_to_text(ids: List[int]) -> str:
    # Keep only 0..127 (ignore BLANK_ID)
    return ''.join(chr(i) for i in ids)

def load_h5py_file(file_path):
    data = {
        'neural_features': [],
        'n_time_steps': [],
        'seq_class_ids': [],
        'seq_len': [],
        'transcriptions': [],
        'sentence_label': [],
        'session': [],
        'block_num': [],
        'trial_num': [],
    }
    # Open the hdf5 file for that day
    with h5py.File(file_path, 'r') as f:

        keys = list(f.keys())

        # For each trial in the selected trials in that day
        for key in keys:
            g = f[key]

            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
    return data
