# Project Name

Brain To Text 2025 Kaggle competition.

## TODOs

For Emma et Cat
- Change Greedy Decoding by a Beam Search Decoding. (TODO in [evaluation.py](evaluation.py))

Later also:
- Try to train with deeper network `(e.g., 8 rnn_layers)`
- We can also try to tak advantage of brain regions segmentation in the model architecture.

## What does the code do (in short):

### Model Description
The code first load the data. Then train a model (CRNN) to predict text from brain signals.
(It is possible to use only a RNN and not CRNN by setting `use_conv=False` in model's arguments)

The model's output is logits. Logits are the raw, unnormalized scores that come out of the last linear layer of your model 
— before any activation like softmax.

Formally, for each time step $t$ and output class $c$:

$$\text{logit}_{t,c} = \mathbf{W}_c^\top \mathbf{h}_t + b_c$$

where:
- $\mathbf{h}_t$ = hidden vector from the encoder (e.g., LSTM or GRU) at time t
- $W_c$, $b_c$ = learned weights and bias for class c

These logits can be any real number (positive, negative, large, small).

The logits are optained by projecting the hidden state of the RNN to the vocabulary size using a linear layer.
`self.proj = nn.Linear(hidden_dim, vocab_size)`

so at each time step:

$$\text{logits}[t, b, c] = \text{model output before softmax}$$

where:
- $t$ = time step
- $b$ batch index
- $c$ = class index (0–127 for ASCII, 128 = blank)
- shape: (T', B, vocab_size)

The vocab_size is 129 (128 ASCII characters + 1 blank for CTC).

### Loss
Once logits predicted, we apply a log_softmax to get log probabilities (CTC loss requires log probabilities as input):

$$\textit{log_probs} = \textit{log_softmax}(\text{logits})$$

Then we use CTC Loss to compute the loss between log_probs and the target text sequences.


## Requirements

- Python 3.12

## Installation

1. Create and activate a virtual environment (macOS example):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Run the main file:
  ```bash
  python main.py
  ```