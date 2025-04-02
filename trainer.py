from collections import defaultdict
import json
import os
from typing import Optional
import attr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parameters import (
    SCHEDULER_DICT,
    HyperParams,
    FIRST_RELEVANT_YEAR,
    LAST_RELEVANT_YEAR,
    FIRST_YEAR,
    MIN_YEARS_BETWEEN_TWO_EVENTS,
    MAX_VALUE_IN_DATA,
)
from models import LossFunctionScheduler, get_model, get_model_like



class Trainer:
    def __init__(self, hyperparams: HyperParams):
        self.hyperparams = hyperparams
        self.latest_model = get_model(hyperparams)
        if hyperparams.start_model_path:
            self.latest_model = ExperimentResults.load(hyperparams.start_model_path).model
        self.best_model = self.latest_model
        self.criterion = LossFunctionScheduler(hyperparams)
        self.optimizer = optim.Adam(self.latest_model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
        self.scheduler = SCHEDULER_DICT[hyperparams.scheduler](self.optimizer, hyperparams.num_epochs, hyperparams.learning_rate)

    @classmethod
    def from_experiment_results(cls, experiment_results: "ExperimentResults"):
        trainer = cls(experiment_results.hyperparams)
        trainer.latest_model = experiment_results.latest_model
        trainer.best_model = experiment_results.model
        return trainer

    def infer(self, X, threshold=0.5):
        with torch.no_grad():
            predictions = self.best_model(X.unsqueeze(0))
            predictions_01 = (predictions.squeeze() >= threshold).int()
            predictions_prob = predictions.squeeze()

        predictions_01[:FIRST_RELEVANT_YEAR - FIRST_YEAR] = 0
        predictions_01[LAST_RELEVANT_YEAR - FIRST_YEAR + 1:] = 0
        pred_idx = (predictions_01 == 1).nonzero(as_tuple=True)[0]
        curr_idx = pred_idx[0] if len(pred_idx) > 0 else None
        for idx in pred_idx[1:]:
            if idx - curr_idx < MIN_YEARS_BETWEEN_TWO_EVENTS:
                if predictions_prob[idx] > predictions_prob[curr_idx]:
                    predictions_01[curr_idx] = 0
                    curr_idx = idx
                else:
                    predictions_01[idx] = 0
            else:
                curr_idx = idx
        return predictions_01, predictions_prob
    
    def infer_to_df(self, dataset, tik_ids, threshold=0.5):
        dfs = []
        for tik_id, (X, _, _) in zip(tik_ids, dataset):
            predictions_01, predictions_prob = self.infer(X, threshold=threshold)
            pos_indices = np.where(predictions_01 == 1)[0]
            for pos_idx in pos_indices:
                dfs.append({
                    'tik_id': tik_id,
                    'year': pos_idx + FIRST_YEAR,
                    'probability': predictions_prob[pos_idx].item()
                })
        return pd.DataFrame(dfs)

    def _evaluate_with_interval(self, reference_y, compared_y, interval_years):
        # get all the indices where y_test is 1
        trues = 0
        falses = 0
        pred_idx = (reference_y == 1).nonzero(as_tuple=True)[0]
        for idx in pred_idx:
            if torch.max(compared_y[max(0, idx-interval_years):idx+interval_years+1]) == 1:
                trues += 1
            else:
                falses += 1
        return trues, falses
    
    def evaluate(self, dataset, plus_minus_years_options=[0, 2, 3], threshold=0.5):
        # Predict
        total_accuracy = 0
        pm_results = {
            i: {
                'TP_precision': 0,
                'FP_precision': 0,
                'TP_recall': 0,
                'FN_recall': 0,
            }
            for i in plus_minus_years_options
        }
        self.best_model.eval()
        for sample in dataset:
            X, y, mask = sample
            mask = mask.bool()
            if not any(mask):
                continue
            predictions_01 = self.infer(X, threshold=threshold)[0]
            total_accuracy += (predictions_01[mask] == y[mask]).float().mean()
            for pm in plus_minus_years_options:
                ts, fs = self._evaluate_with_interval(predictions_01[mask], y[mask], pm)
                pm_results[pm]['TP_precision'] += ts
                pm_results[pm]['FP_precision'] += fs
                ts, fs = self._evaluate_with_interval(y[mask], predictions_01[mask], pm)
                pm_results[pm]['TP_recall'] += ts
                pm_results[pm]['FN_recall'] += fs
                
        total_accuracy /= len(dataset)
        scores = {'accuracy': float(total_accuracy)}
        for pm in plus_minus_years_options:
            res = pm_results[pm]
            recall = res['TP_recall'] / (res['TP_recall'] + res['FN_recall']) if (res['TP_recall'] + res['FN_recall']) > 0 else 0
            precision = res['TP_precision'] / (res['TP_precision'] + res['FP_precision']) if (res['TP_precision'] + res['FP_precision']) > 0 else 0
            scores[f'recall_pm{pm}'] = recall
            scores[f'precision_pm{pm}'] = precision
            scores[f'f1_pm{pm}'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return scores
    
    def train(self, dataset_train, dataset_val, verbose=True, show_fig=False) -> pd.DataFrame:
        losses = []
        dataloader = DataLoader(dataset_train, batch_size=self.hyperparams.batch_size, shuffle=True)
        fig, ax = None, None
        best_score = -1
        best_state_dict = None  # Store the state dict instead of the model
        
        for epoch in tqdm(range(self.hyperparams.num_epochs)):
            self.latest_model.train()
            epoch_losses = []
            for sample in dataloader:
                X_batch, y_batch, mask = sample
                if self.hyperparams.normalize_data:
                    X_batch = X_batch / MAX_VALUE_IN_DATA
                self.optimizer.zero_grad()
                outputs = self.latest_model(X_batch)
                loss = self.criterion.loss_fn(outputs.squeeze(), y_batch, mask)
                loss.backward()
                if self.hyperparams.apply_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.latest_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())
            
            self.criterion.step()
            self.scheduler.step()
            losses.append({
                'epoch': epoch,
                'type': 'loss',
                'value': sum(epoch_losses) / len(epoch_losses)
            })
            if epoch % self.hyperparams.eval_every_n_epochs == 0:
                last_loss = losses[-1]['value']
                ev_train = self.evaluate(dataset_train)
                ev_val = self.evaluate(dataset_val)
                for k, v in ev_train.items():
                    losses.append({
                        'epoch': epoch,
                        'type': f'train_{k}',
                        'value': v
                    })
                for k, v in ev_val.items():
                    losses.append({
                        'epoch': epoch,
                        'type': f'val_{k}',
                        'value': v
                    })
                if self.hyperparams.use_max_metric:
                    if ev_val.get(self.hyperparams.use_max_metric, -2) > best_score:
                        best_score = ev_val.get(self.hyperparams.use_max_metric, -2)
                        # Save a deep copy of the model's state
                        best_state_dict = {k: v.cpu().clone() for k, v in self.latest_model.state_dict().items()}
                if verbose:
                    ev_train_text = ', '.join([f'{k}: {v:.4f}' for k, v in ev_train.items()])
                    ev_test_text = ', '.join([f'{k}: {v:.4f}' for k, v in ev_val.items()])
                    tqdm.write(f"Epoch [{epoch+1}/{self.hyperparams.num_epochs}], Loss: {last_loss:.4f}, Train: {ev_train_text}, Val: {ev_test_text}")
                if show_fig:
                    fig, ax = ExperimentResults.plot_losses_from_dict(losses, fig, ax)
        
        # At the end of training, load the best model if we were tracking it
        if best_state_dict is not None:
            self.best_model = get_model_like(self.latest_model)
            self.best_model.load_state_dict(best_state_dict)
        else:
            self.best_model = self.latest_model
        return pd.DataFrame(losses)
    
    def get_precision_recall_curve(self, dataset, plus_minus_years_options=[0, 2, 3], min_probability_diff=1e-5) -> pd.DataFrame:
        dfs = defaultdict(list)
        for X, y, mask in dataset:
            predictions_prob = self.infer(X)[1].numpy()
            pos_indices = np.where(y.bool() & mask.bool())[0]
            for pm in plus_minus_years_options:
                pos_probs = np.array([max(predictions_prob[max(0, i - pm): i + pm + 1]) for i in pos_indices])
                kernel = np.ones(pm * 2 + 1)
                neg_mask = (1 - np.convolve(y.bool(), kernel, mode='same')).astype(bool) & mask.bool().numpy()
                neg_probs = predictions_prob[neg_mask]
                df = pd.DataFrame({
                    'probability': np.concatenate([pos_probs, neg_probs]),
                    'label': np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))]),
                    'plus_minus_years': pm
                })
                dfs[pm].append(df)
        pr_curves = []
        for pm in plus_minus_years_options:
            df_probs = pd.concat(dfs[pm])
            df_probs = df_probs.sort_values('probability', ascending=False, ignore_index=True)
            df_probs['cumsum'] = df_probs['label'].cumsum()
            df_probs['precision'] = df_probs['cumsum'] / (df_probs.index + 1)
            df_probs['recall'] = df_probs['cumsum'] / df_probs['label'].sum()
            # Combine similar probabilities
            for ind in range(len(df_probs) - 1):
                if abs(df_probs.loc[ind]['probability'] - df_probs.loc[ind+1]['probability']) < min_probability_diff:
                    df_probs = df_probs.drop(index=ind)
            # Remove the last row, as it's 1
            df_probs = df_probs.drop(index=df_probs.index[-1])
            pr_curves.append(df_probs)
        return pd.concat(pr_curves)
    
    def plot_naive_pr_curve(self, dataset, plus_minus_years_options=[0, 3], num_bins=100):
        thresholds = np.linspace(0, 1, num_bins)
        dfs = []
        for threshold in thresholds:
            eval = self.evaluate(dataset, threshold=threshold, plus_minus_years_options=plus_minus_years_options)
            eval['threshold'] = threshold
            dfs.append(eval)
        df = pd.DataFrame(dfs)
        for pm in plus_minus_years_options:
            sns.lineplot(data=df, x=f'recall_pm{pm}', y=f'precision_pm{pm}', marker='o', label=f'pm={pm}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        return df


@attr.s(auto_attribs=True)
class ExperimentResults:
    hyperparams: HyperParams
    model: nn.Module
    losses: pd.DataFrame
    precision_recall_curve: pd.DataFrame
    latest_model: Optional[nn.Module] = None

    def save(self, path: Optional[str] = None):
        path = path or '.'
        full_path = os.path.join(path, f'{self.hyperparams.experiment_name}')
        os.makedirs(full_path, exist_ok=True)
        with open(os.path.join(full_path, 'hyperparams.json'), 'w') as f:
            json.dump(self.hyperparams.to_dict(), f)
        torch.save(self.model.state_dict(), os.path.join(full_path, 'model.pth'))
        if self.latest_model is not None:
            torch.save(self.latest_model.state_dict(), os.path.join(full_path, 'latest_model.pth'))
        self.losses.to_csv(os.path.join(full_path, 'losses.csv'), index=False)
        self.precision_recall_curve.to_csv(os.path.join(full_path, 'precision_recall_curve.csv'), index=False)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, 'hyperparams.json'), 'r') as f:
            hyperparams = HyperParams.from_dict(json.load(f))
        model = get_model(hyperparams)
        model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
        latest_model_path = os.path.join(path, 'latest_model.pth')
        if os.path.exists(latest_model_path):
            latest_model = get_model_like(model)
            latest_model.load_state_dict(torch.load(latest_model_path))
        else:
            latest_model = None
        losses = pd.read_csv(os.path.join(path, 'losses.csv'))
        precision_recall_curve = pd.read_csv(os.path.join(path, 'precision_recall_curve.csv'))
        return cls(hyperparams, model, losses, precision_recall_curve, latest_model)

    def plot_losses(self, fig=None, ax=None) -> tuple[plt.Figure, plt.Axes]:
        # Create new figure and axes if not provided
        if fig is None or ax is None:
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        else:
            # Clear existing axes for updating
            for a in ax.ravel():
                a.clear()
        losses0 = self.losses[self.losses.type == 'loss']
        losses1 = self.losses[self.losses.type.map(lambda x: 'accuracy' in x)]
        losses2 = self.losses[self.losses.type.map(lambda x: 'precision' in x)]
        losses3 = self.losses[self.losses.type.map(lambda x: 'recall' in x)]
        sns.lineplot(data=losses0, x='epoch', y='value', hue='type', ax=ax[0, 0])
        sns.lineplot(data=losses1, x='epoch', y='value', hue='type', ax=ax[0, 1])
        sns.lineplot(data=losses2, x='epoch', y='value', hue='type', ax=ax[1, 0])
        sns.lineplot(data=losses3, x='epoch', y='value', hue='type', ax=ax[1, 1])
        ax[0, 0].set_title('Loss')
        ax[0, 1].set_title('Accuracy')
        ax[1, 0].set_title('Precision')
        ax[1, 1].set_title('Recall')
        for a in ax.ravel():
            a.legend(loc='best')
        fig.tight_layout()
        plt.show()
        return fig, ax

    @classmethod
    def plot_losses_from_dict(cls, losses: list[dict] | pd.DataFrame, fig = None, ax = None) -> tuple[plt.Figure, plt.Axes]:
        return cls(None, None, pd.DataFrame(losses), None).plot_losses(fig, ax)

    def plot_precision_recall_curve(self):
        sns.lineplot(data=self.precision_recall_curve, x='recall', y='precision', hue='plus_minus_years')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

    @classmethod
    def get_all_experiments_names(cls, experiments_folder: str):
        return [f for f in os.listdir(experiments_folder) if os.path.isdir(os.path.join(experiments_folder, f))]
    
    @classmethod
    def plot_multiple_pr_curves(cls, experiments_folder: str, experiment_names: Optional[list[str]] = None, plus_minus_years: list[int] = [3]):
        if experiment_names is None:
            experiment_names = cls.get_all_experiments_names(experiments_folder)
        fig, ax = plt.subplots(figsize=(10, 5))
        for experiment_name in experiment_names:
            experiment = cls.load(os.path.join(experiments_folder, experiment_name))
            for pm in plus_minus_years:
                sns.lineplot(data=experiment.precision_recall_curve[experiment.precision_recall_curve.plus_minus_years == pm],
                              x='recall', y='precision', marker='o', label=f'{experiment_name} {pm=}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
