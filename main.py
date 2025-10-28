import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for MPS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from load_data import load_h5py_file
from tqdm import tqdm


from data import NeuralDataset, collate_batch
from evaluation import run_evaluate
from models import CTCEncoder, ConformerCTC
from training import Trainer, EarlyStopping


# ---------------------------
# Device configuration
# ---------------------------
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
print(f"Using device: {DEVICE}")

# ---------------------------
# Vocab / constants
# ---------------------------
# Vocabulary: A-Z, a-z, space, period, comma, question mark, exclamation mark
ASCII_SIZE = 128  # ASCII 0..127
BLANK_ID = 128         # 128 # CTC blank token id
VOCAB_SIZE = ASCII_SIZE + 1  # output vocab size (including blank)
FEATURE_LEN = 512

# Neural region blocks (type_id, region_id) -> (start, end)
BLOCKS = {
    # (type_id, region_id): (start, end)  # end exclusive
    (0, 0): (0,   64),    # TC ventral6v
    (0, 1): (64,  128),   # TC area4
    (0, 2): (128, 192),   # TC 55b
    (0, 3): (192, 256),   # TC dorsal6v
    (1, 0): (256, 320),   # SBP ventral6v
    (1, 1): (320, 384),   # SBP area4
    (1, 2): (384, 448),   # SBP 55b
    (1, 3): (448, 512),   # SBP dorsal6v
}


def main(debug=False, train=True):
    # ------------------------ Load data from hdf5 files to Dataframe ------------------------
    path = "./brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final/"

    folders = os.listdir(path)
    train_files = []
    val_files = []
    for i, files in enumerate(folders):
        if files.startswith("."):
            continue
        files = os.listdir(os.path.join(path, files))
        for file in files:
            if file.endswith("train.hdf5"):
                train_files.append(os.path.join(path, folders[i], file))
            elif file.endswith("val.hdf5"):
                val_files.append(os.path.join(path, folders[i], file))

    train_df = pd.DataFrame()
    i = 0
    for file in tqdm(train_files, desc="Loading train files"):
        data = load_h5py_file(file)
        temp_df = pd.DataFrame(data)
        train_df = pd.concat([train_df, temp_df], ignore_index=True)
        if debug:
            i += 1
            if i >= 4:  # load only 4 files in debug mode
                break

    val_df = pd.DataFrame()
    i = 0
    for file in tqdm(val_files, desc="Loading val files"):
        data = load_h5py_file(file)
        temp_df = pd.DataFrame(data)
        val_df = pd.concat([val_df, temp_df], ignore_index=True)
        if debug:
            i += 1
            if i >= 2:  # load only 2 files in debug mode
                break


    # ------------------------ Dataset and Dataloader ------------------------
    train_dataset = NeuralDataset(train_df, blank_id=BLANK_ID)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b)
    )

    val_dataset = NeuralDataset(val_df, blank_id=BLANK_ID)
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b)
    )

    # ------------------------ Define Model ------------------------
    # We only use 2 layers for fast training. Increase number of layers for better accuracy.
    model = CTCEncoder(
        vocab_size=VOCAB_SIZE,
        blank_id=BLANK_ID,
        rnn_layers=2,
        use_gru=True
    ).to(DEVICE)

    # number of model parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} total parameters.")
    print(f"Model has {num_trainable_params:,} trainable parameters.")

    # ------------------------ Training ------------------------
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=1e-3),
        loss_fn=nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True),
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=100,
        blank_id=BLANK_ID,
        early_stop=EarlyStopping(patience=5, min_delta=1e-3, path=f"./model/{model.__str__()}_best_model.pt"),
        sample_interval=5,
    )
    if train:
        print("Starting training...")
        trainer.run()
    else:
        model.load_state_dict(torch.load(f"./model/{model.__str__()}_best_model.pt", map_location=DEVICE))
    trainer.predict_sample()

    # ------------------------ Evaluation on Test Set ------------------------
    test_cer, test_wer = run_evaluate(model, val_loader, blank_id=BLANK_ID, device=DEVICE)


if __name__ == '__main__':
    main(debug=True, train=True)