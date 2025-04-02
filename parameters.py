from dataclasses import dataclass, asdict
from typing import Optional
import torch.optim as optim


TRAIN_PERCENT = 80
VAL_PERCENT = 10
FIRST_RELEVANT_YEAR = 1974
LAST_RELEVANT_YEAR = 1997
INTERVAL_YEARS = 5
FIRST_YEAR = FIRST_RELEVANT_YEAR - INTERVAL_YEARS
LAST_YEAR = LAST_RELEVANT_YEAR + INTERVAL_YEARS
NUM_FIELDS = 68
MIN_YEARS_BETWEEN_TWO_EVENTS = 15
FIELDS = [f'_f{i + 1}' for i in range(NUM_FIELDS)]
MAX_VALUE_IN_DATA = 150


def fuzzy_scheduler(optimizer, num_steps, max_lr=0.01):
    onecycle_steps = 4000
    cosine_steps = 10000
    
    scheduler_onecycle = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=onecycle_steps,
        pct_start=0.3,
        cycle_momentum=False,
        anneal_strategy='linear'
    )

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,  # Maximum number of iterations
        eta_min=0.0001  # Minimum learning rate
    )

    scheduler_step = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=100,
        gamma=0.01,
    )

    # Chain schedulers sequentially
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_onecycle, scheduler_cosine, scheduler_step],
        milestones=[onecycle_steps, onecycle_steps + cosine_steps]
    )
    return scheduler


def onecycle_scheduler(optimizer, num_steps, max_lr=0.01):
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=num_steps, pct_start=0.3, cycle_momentum=False, anneal_strategy='linear')


def halfcycle_scheduler(optimizer, num_steps, max_lr=0.01):
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=num_steps*4, pct_start=0.3, cycle_momentum=False, anneal_strategy='linear')


def cosine_scheduler(optimizer, num_steps, max_lr=0.01):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0.0001)


SCHEDULER_DICT = {
    'fuzzy': fuzzy_scheduler,
    'onecycle': onecycle_scheduler,
    'cosine': cosine_scheduler,
    'halfcycle': halfcycle_scheduler,
}


@dataclass
class PosWeightSchedule:
    pos_weight: float
    num_epochs: int

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass
class HyperParams:
    experiment_name: str  # The name of the experiment
    input_size: int = NUM_FIELDS  # The number of input features
    total_years: int = LAST_YEAR - FIRST_YEAR + 1  # The number of years in the dataset
    bidirectional: bool = True  # Whether the RNN is bidirectional
    num_epochs: int = 15001  # The number of epochs to train for
    pos_weight: float = 100  # The weight of the positive class in the loss function
    pos_weight_schedule: Optional[list[PosWeightSchedule]] = None  # The schedule for the pos_weight, if set, overrides pos_weight
    learning_rate: float = 0.01  # The learning rate
    scheduler: str = 'fuzzy'  # The scheduler to use
    batch_size: int = 32  # The batch size
    eval_every_n_epochs: int = 100  # The number of epochs to evaluate the model
    start_model_path: Optional[str] = None  # path to the model to start from
    use_max_metric: Optional[str] = None  # The name of the metric that we return the model with its highest value
    dropout: float = 0.0  # The dropout rate for the model
    weight_decay: float = 0.0  # The weight decay for the optimizer
    loss_function: str = 'bce_with_mask'  # The loss function to use
    model_type: str = 'rnn'  # The type of model to use
    apply_gradient_clipping: bool = False  # Whether to apply gradient clipping
    normalize_data: bool = False  # Whether to normalize the data
    # RNN info
    hidden_size: int = 5  # The number of hidden units in the RNN
    num_rnn_layers: int = 1  # The number of RNN layers
    rnn_use_tanh_instead_of_relu: bool = False  # Whether to use tanh instead of relu in the RNN
    # LSTM info
    num_lstm_layers: int = 1  # The number of LSTM layers
    # Transformer info
    nhead: int = 1  # The number of attention heads in the Transformer
    num_transformer_layers: int = 1  # The number of Transformer layers
    dim_feedforward: int = 5  # The dimension of the feedforward network in the Transformer

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict):
        ret = cls(**d)
        if d.get('pos_weight_schedule'):
            ret.pos_weight_schedule = [PosWeightSchedule.from_dict(pws) for pws in d['pos_weight_schedule']]
        return ret
