# =====================
# File: src/data_loading.py
# =====================
import pandas as pd
import numpy as np
import torch


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Undersample: 4x the number of label=1 businesses
    bizid_counts = df.groupby(['bizId', 'label']).size().reset_index(name='counts')
    unique_bizid_label_1_count = bizid_counts[bizid_counts['label'] == 1]['bizId'].nunique()
    num_top_bizids_label_0 = unique_bizid_label_1_count * 4
    bizids_label_0 = bizid_counts[bizid_counts['label'] == 0]
    top_bizids_label_0 = bizids_label_0.nlargest(num_top_bizids_label_0, 'counts')['bizId'].unique()
    df = df[df['bizId'].isin(top_bizids_label_0) | (df['label'].isin([1]))]

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def index_columns(df):
    def indexing(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    rev2idx = indexing(df['review_vector'])
    biz2idx = indexing(df['bizId'])
    reviewer2idx = indexing(df['authorName'])

    df['revIdx'] = df['review_vector'].map(rev2idx)
    df['bizIdx'] = df['bizId'].map(biz2idx)
    df['reviewerIdx'] = df['authorName'].map(reviewer2idx)

    return df, rev2idx, biz2idx, reviewer2idx
