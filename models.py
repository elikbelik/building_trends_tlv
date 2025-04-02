import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from parameters import NUM_FIELDS, HyperParams, PosWeightSchedule


class BCEWithLogitsLossWithMask(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight=None):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        super(BCEWithLogitsLossWithMask, self).__init__(reduction='none', pos_weight=pos_weight)
        self.pos_weight = pos_weight if pos_weight is not None else 1.0

    def forward(self, y_pred, y, mask):
        y_prob = 1 / (1 + torch.exp(-y_pred))
        loss = -(self.pos_weight * y * torch.log(y_prob + 1e-8) + (1 - y) * torch.log(1 - y_prob + 1e-8))
        masked_loss = loss * mask
        if mask.sum() == 0:
            return masked_loss.sum()
        return masked_loss.sum() / mask.sum()
    

class BCEWithLogitsLossWithMaskWithPM(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight=None, plus_minus_years=0):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
        super(BCEWithLogitsLossWithMaskWithPM, self).__init__(reduction='none', pos_weight=pos_weight)
        self.pos_weight = pos_weight if pos_weight is not None else 1.0
        self.plus_minus_years = plus_minus_years

    def forward(self, y_pred, y, mask):
        y_prob = 1 / (1 + torch.exp(-y_pred))
        batch_size = y_prob.shape[0]
        total_loss = 0
        for i in range(batch_size):
            pos_indices = (y[i] == 1).nonzero(as_tuple=True)[0]
            mask_clone = mask[i].clone()
            for idx in pos_indices:
                mask_clone[max(0, idx-self.plus_minus_years):idx+self.plus_minus_years+1] = 0
                max_ind = torch.argmax(y_prob[i, max(0, idx-self.plus_minus_years):idx+self.plus_minus_years+1])
                mask_clone[max_ind] = 1
            loss = -(self.pos_weight * y[i] * torch.log(y_prob[i] + 1e-8) + (1 - y[i]) * torch.log(1 - y_prob[i] + 1e-8))
            masked_loss = loss * mask_clone
            if mask_clone.sum() == 0:
                total_loss += masked_loss.sum()
            else:
                total_loss += masked_loss.sum() / mask_clone.sum()
        return total_loss


def get_loss_function(hyperparams: HyperParams, pos_weight: Optional[float] = None):
    pos_weight = pos_weight or hyperparams.pos_weight
    if hyperparams.loss_function == 'bce_with_mask':
        return BCEWithLogitsLossWithMask(pos_weight=pos_weight)
    elif hyperparams.loss_function == 'bce_with_mask_with_pm3':
        return BCEWithLogitsLossWithMaskWithPM(pos_weight=pos_weight, plus_minus_years=3)
    else:
        raise ValueError(f'Unknown loss function: {hyperparams.loss_function}')


class LossFunctionScheduler:
    def __init__(self, hyperparams: HyperParams, verbose: bool = True):
        self.hyperparams = hyperparams
        self.loss_fn_name = hyperparams.loss_function
        self.schedules: list[PosWeightSchedule] = hyperparams.pos_weight_schedule or []
        self.default_pos_weight = hyperparams.pos_weight
        self.curr_idx = 0
        self.verbose = verbose
        self._next_epoch_to_update = self.schedules[0].num_epochs if self.schedules else -1
        self.loss_fn = get_loss_function(hyperparams, self.get_current_pos_weight())

    def get_current_pos_weight(self) -> float:
        pw = self.schedules[self.curr_idx].pos_weight if self.curr_idx < len(self.schedules) else self.default_pos_weight
        if self.verbose:
            print(f'Current pos weight: {pw}')
        return pw

    def step(self):
        if self._next_epoch_to_update > 0:
            self._next_epoch_to_update -= 1
        if self._next_epoch_to_update == 0:
            self.curr_idx += 1
            self._next_epoch_to_update = self.schedules[self.curr_idx].num_epochs if self.curr_idx < len(self.schedules) else -1
            self.loss_fn = get_loss_function(self.hyperparams, self.get_current_pos_weight())


class BinaryClassificationRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool,
                 num_rnn_layers: int,
                 rnn_dropout: float,
                 use_tanh_instead_of_relu: bool = False):
        super(BinaryClassificationRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_rnn_layers = num_rnn_layers
        self.rnn_dropout = rnn_dropout
        self.use_tanh_instead_of_relu = use_tanh_instead_of_relu
        nonlinearity = 'tanh' if use_tanh_instead_of_relu else 'relu'
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_rnn_layers, batch_first=True,
                          bidirectional=bidirectional, nonlinearity=nonlinearity, dropout=rnn_dropout)
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), 1)
        self.final = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.final(out)
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    @classmethod
    def like(cls, other: 'BinaryClassificationRNN'):
        return cls(other.input_size, other.hidden_size, other.bidirectional, other.num_rnn_layers, other.rnn_dropout, other.use_tanh_instead_of_relu)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=NUM_FIELDS):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)  # Add positional encoding
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.input_projection = nn.Linear(input_dim, d_model)  # Project input to model dimension
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction Layer
        self.classifier = nn.Linear(d_model, 1)  # Output single value (binary classification)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)  # Project input to d_model
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through Transformer encoder
        
        logits = self.classifier(x).squeeze(-1)  # Predict label for each time point
        return torch.sigmoid(logits)  # Apply sigmoid for binary classification probabilities

    @classmethod
    def like(cls, other: 'TimeSeriesTransformer'):
        return cls(other.input_dim, other.d_model, other.nhead, other.num_layers, other.dim_feedforward, other.dropout)
    

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional: bool, dropout: float):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), 1)
        self.final = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size * (1 + bidirectional))
        out = self.fc(lstm_out)  # out shape: (batch_size, seq_len, 1)
        out = self.final(out)
        return out
    
    @classmethod
    def like(cls, other: 'BiLSTM'):
        return cls(other.input_size, other.hidden_size, other.num_layers, other.bidirectional, other.dropout)
    

def get_model(hyperparams: HyperParams):
    if hyperparams.model_type == 'rnn':
        return BinaryClassificationRNN(
            input_size=hyperparams.input_size,
            hidden_size=hyperparams.hidden_size,
            bidirectional=hyperparams.bidirectional,
            num_rnn_layers=hyperparams.num_rnn_layers,
            rnn_dropout=hyperparams.dropout,
            use_tanh_instead_of_relu=hyperparams.rnn_use_tanh_instead_of_relu
        )
    elif hyperparams.model_type == 'transformer':
        return TimeSeriesTransformer(
            input_dim=hyperparams.input_size,
            d_model=hyperparams.total_years,
            nhead=hyperparams.nhead,
            num_layers=hyperparams.num_transformer_layers,
            dim_feedforward=hyperparams.dim_feedforward,
            dropout=hyperparams.dropout
        )
    elif hyperparams.model_type == 'lstm':
        return BiLSTM(
            input_size=hyperparams.input_size,
            hidden_size=hyperparams.hidden_size,
            num_layers=hyperparams.num_lstm_layers,
            bidirectional=hyperparams.bidirectional,
            dropout=hyperparams.dropout
        )
    else:
        raise ValueError(f'Unknown model type: {hyperparams.model_type}')


def get_model_like(model: nn.Module):
    if isinstance(model, BinaryClassificationRNN):
        return BinaryClassificationRNN.like(model)
    elif isinstance(model, TimeSeriesTransformer):
        return TimeSeriesTransformer.like(model)
    elif isinstance(model, BiLSTM):
        return BiLSTM.like(model)
    else:
        raise ValueError(f'Unknown model type: {type(model)}')
