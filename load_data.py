import pandas as pd
from collections import defaultdict
from tqdm.notebook import tqdm
import hashlib
import torch
import numpy as np
from typing import Tuple

from parameters import (
    TRAIN_PERCENT,
    FIRST_RELEVANT_YEAR,
    LAST_RELEVANT_YEAR,
    INTERVAL_YEARS,
    FIRST_YEAR,
    LAST_YEAR,
    NUM_FIELDS,
    FIELDS,
    VAL_PERCENT,
)

TensorTuple3 = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

tqdm.pandas()


DATA_CSV = 'data/pivoted_data.csv'
MINOR_MAJOR_CSV = 'data/cleaned74-97_validated_updated_fixed.csv'
MINOR_MAJOR_PART2_CSV = 'data/machine learning 4-1 validation_EH.csv'
MINOR_MAJOR_PART3_CSV = 'data/ml_new_entress.csv'
MINOR_MAJOR_PART4_70S_CSV = 'data/69-79 validation.csv'


def stable_hash(value):
    """Compute a stable hash for the given value."""
    value_str = str(value).encode('utf-8')  # Convert to string and then bytes
    hash_object = hashlib.md5(value_str)  # Use MD5 or other algorithms
    return int(hash_object.hexdigest(), 16)  # Convert hash to an integer


def rec_to_vec(rec):
    vec_list = defaultdict(lambda: [0]*NUM_FIELDS)
    for f_idx, field in enumerate(FIELDS):
        if not isinstance(rec[field], str):
            continue
        year_list = eval(rec[field])
        for year in year_list:
            if year < FIRST_YEAR:
                continue
            vec_list[year][f_idx] += 1
    return vec_list


def load_unlabeled_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_CSV)
    data.drop(columns=['true_year'], inplace=True)
    data.drop_duplicates(['tik_id'], inplace=True)

    data['fold_id'] = data.tik_id.apply(stable_hash) % 10
    data['fold'] = data.fold_id.map(lambda x: 'train' if x < TRAIN_PERCENT/10 else 'val' if x < TRAIN_PERCENT/10 + VAL_PERCENT/10 else 'test')
    data['min_year'] = data.apply(lambda rec: min([min(eval(v)) for x, v in rec.items() if x.startswith('_f') and isinstance(v, str)]), axis=1)
    data['max_year'] = data.apply(lambda rec: max([max(eval(v)) for x, v in rec.items() if x.startswith('_f') and isinstance(v, str)]), axis=1)
    data['year_diff'] = data.max_year - data.min_year
    data['vec'] = data.progress_apply(rec_to_vec, axis=1)
    return data


def load_data(add_part3: bool = False, add_part4_70s: bool = False) -> pd.DataFrame:
    data = load_unlabeled_data()
    # Combine with minor major data
    minor_major = pd.read_csv(MINOR_MAJOR_CSV)
    minor_major['label'] = minor_major.YEAR
    minor_major.loc[minor_major.MAJOR.isna(), 'label'] = None
    minor_major.drop_duplicates('tik_id', keep='first', inplace=True)
    if add_part3:
        minor_major_part3 = pd.read_csv(MINOR_MAJOR_PART3_CSV)
        minor_major_part3['label'] = minor_major_part3.YEAR
        minor_major_part3.loc[minor_major_part3.YEAR.isna(), 'label'] = None
        minor_major = pd.concat([minor_major, minor_major_part3])
        minor_major.drop_duplicates('tik_id', keep='first', inplace=True)
    if add_part4_70s:
        minor_major_part4_70s = pd.read_csv(MINOR_MAJOR_PART4_70S_CSV)
        minor_major_part4_70s.drop_duplicates('tik_id', keep='first', inplace=True)
        minor_major_part4_70s.YEAR = pd.to_numeric(minor_major_part4_70s.YEAR, errors='coerce')
        minor_major_part4_70s['label'] = minor_major_part4_70s.YEAR
        minor_major_part4_70s.loc[minor_major_part4_70s.YEAR.isna(), 'label'] = None
        minor_major = pd.concat([minor_major, minor_major_part4_70s])
        minor_major.drop_duplicates('tik_id', keep='last', inplace=True)

    minor_major_part2 = pd.read_csv(MINOR_MAJOR_PART2_CSV)
    minor_major_part2['YEAR'] = minor_major_part2.estimated_year
    minor_major_part2['other_year'] = minor_major_part2['other year?']
    minor_major_part2_filtered = minor_major_part2[(minor_major_part2.Major.notna()) & (minor_major_part2.estimated_year >= FIRST_RELEVANT_YEAR) & (minor_major_part2.estimated_year <= LAST_RELEVANT_YEAR)]
    labeled_data = data.merge(minor_major, on='tik_id', how='inner')
    labeled_data['know_all_years'] = False
    known_years = minor_major_part2_filtered[minor_major_part2_filtered.other_year.notna()].tik_id
    if add_part3:
        known_years = pd.concat([known_years, minor_major_part3.tik_id])
    if add_part4_70s:
        known_years = pd.concat([known_years, minor_major_part4_70s.tik_id])
        
    labeled_data.loc[labeled_data.tik_id.isin(known_years), 'know_all_years'] = True
    return labeled_data


def _get_record_x_only(rec) -> TensorTuple3:
    vec = rec.vec
    max_years = LAST_YEAR - FIRST_YEAR
    X = torch.tensor([vec[FIRST_YEAR + year] for year in range(max_years+1)], dtype=torch.float32)
    return X, None, None


def _get_record_x_y_with_mask(rec) -> TensorTuple3:
    vec = rec.vec
    max_years = LAST_YEAR - FIRST_YEAR
    X = torch.tensor([vec[FIRST_YEAR + year] for year in range(max_years+1)], dtype=torch.float32)
    y = torch.zeros(max_years+1, dtype=torch.float32)
    mask = torch.zeros(max_years+1, dtype=torch.float32)
    if rec.label is not None and not np.isnan(rec.label):
        if FIRST_RELEVANT_YEAR <= rec.label <= LAST_RELEVANT_YEAR:
            y[int(rec.label - FIRST_YEAR)] = 1
        other_years = rec.additional_year
        if FIRST_RELEVANT_YEAR <= other_years <= LAST_RELEVANT_YEAR:
            y[int(other_years - FIRST_YEAR)] = 1
        mask[INTERVAL_YEARS: -INTERVAL_YEARS] = 1
    elif rec.know_all_years:
        mask[INTERVAL_YEARS: -INTERVAL_YEARS] = 1
    else:
        mask[INTERVAL_YEARS: int(rec.YEAR - FIRST_YEAR) + 1] = 1
    return X, y, mask


def create_dataset(data: pd.DataFrame) -> Tuple[list[TensorTuple3], list[TensorTuple3], list[TensorTuple3]]:
    train_data = data[data.fold == 'train']
    val_data = data[data.fold == 'val']
    test_data = data[data.fold == 'test']
    dataset_train = [_get_record_x_y_with_mask(rec) for rec in train_data.itertuples()]
    dataset_val = [_get_record_x_y_with_mask(rec) for rec in val_data.itertuples()]
    dataset_test = [_get_record_x_y_with_mask(rec) for rec in test_data.itertuples()]
    return dataset_train, dataset_val, dataset_test


def get_dataset_tik_ids(data: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    train_data = data[data.fold == 'train']
    val_data = data[data.fold == 'val']
    test_data = data[data.fold == 'test']
    return train_data.tik_id.tolist(), val_data.tik_id.tolist(), test_data.tik_id.tolist()


def create_unlabeled_dataset(data: pd.DataFrame) -> Tuple[list[TensorTuple3], list[str]]:
    return [_get_record_x_only(rec) for rec in data.itertuples()], data.tik_id.tolist()
