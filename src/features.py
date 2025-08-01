# =====================
# File: src/features.py
# =====================
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch


def get_business_features(df):
    features = [
        'yelp_feat1'
        'yelp_feat2'
        'yelp_feat3'
        'yelp_feat4'
        'yelp_feat5'
        'yelp_feat6'
        'yelp_feat7'
        'yelp_feat8'
        'yelp_feat9'
        'yelp_feat10'
        'yelp_feat11'
        'yelp_feat12'
        'licence_feat1'
        'gis_feat1'
        'gis_feat2'
        'gis_feat3'
        'census_feat1'
        'census_feat2'
        'census_feat3'
        'census_feat4'
        'census_feat5'
        'census_feat6'
        'census_feat7'
        'landcover_feat1'
    ]
    df_sorted = df.drop_duplicates(subset='bizIdx').sort_values('bizIdx')
    return df_sorted[features].values


def get_review_features(df):
    df_sorted = df.drop_duplicates(subset='revIdx').sort_values('revIdx')
    embeddings = np.array(df_sorted['review_feat1'].apply(lambda x: [float(val) for val in x.strip('[]').split()]).tolist())
    pca = PCA(n_components=22)
    reduced_embeddings = pca.fit_transform(embeddings)
    cols1 = ['roberta_feat1', 'roberta_feat2', 'roberta_feat3']
    cols2 = ['review_feat2', 'lexicon_feat1', 'lexicon_feat2']
    X_review = pd.concat([
        pd.DataFrame(reduced_embeddings),
        df_sorted[cols1].reset_index(drop=True),
        df_sorted[cols2].reset_index(drop=True)
    ], axis=1)
    return X_review.values


def get_reviewer_features(df):
    df_sorted = df.drop_duplicates(subset='reviewerIdx').sort_values('reviewerIdx')
    encoded = pd.get_dummies(df_sorted[['author_feat2']], columns=['author_feat2'], prefix='gender')
    encoded = encoded.astype(int)
    for i in range(1, 23):
        encoded[f'random_col_{i}'] = np.random.uniform(-0.5, 0.5, len(encoded))
    return encoded.values
