# =====================
# File: src/features.py
# =====================
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch


def get_business_features(df):
    features = [
        'yelp_close9',
        'yelp_close10',
        'yelp_avg_rating_moreThan4',
        'yelp_reviewRating_min_is5',
        'yelp_avg_rating_lessThan2',
        'yelp_massageCat',
        'yelp_business_name_combine',
        'yelp_spaCat',
        'yelp_phone_advertisement',
        'census_pct_manufacturing_industry_low',
        'landcover_type_developed_high_intensity',
        'yelp_category_reflexology',
        'yelp_reviewRating_max_is1',
        'yelp_lexicon_score_mean_high',
        'census_pct_nonwhite_high',
        'census_avg_household_size_high',
        'min_dist_base_high',
        'census_pct_20_to_29_low',
        'yelp_authorGender_pct_male_zero',
        'yelp_close11',
        'yelp_authorGender_pct_male_high',
        'min_dist_police_low',
        'census_pct_housing_vacant_low',
        'census_pct_households_with_children_low',
        'yelp_lexicon_score_mean_zero',
        'census_pct_over25_with_bachelors_low',
        'min_dist_base_low',
        'owner_listed_worker_out_of_state'
    ]
    df_sorted = df.drop_duplicates(subset='bizIdx').sort_values('bizIdx')
    return df_sorted[features].values


def get_review_features(df):
    df_sorted = df.drop_duplicates(subset='revIdx').sort_values('revIdx')
    embeddings = np.array(df_sorted['review_vector'].apply(lambda x: [float(val) for val in x.strip('[]').split()]).tolist())
    pca = PCA(n_components=22)
    reduced_embeddings = pca.fit_transform(embeddings)
    cols1 = ['roberta_neu', 'roberta_pos', 'roberta_neg']
    cols2 = ['reviewRating', 'lexicon_score', 'lexicon_prediction']
    X_review = pd.concat([
        pd.DataFrame(reduced_embeddings),
        df_sorted[cols1].reset_index(drop=True),
        df_sorted[cols2].reset_index(drop=True)
    ], axis=1)
    return X_review.values


def get_reviewer_features(df):
    df_sorted = df.drop_duplicates(subset='reviewerIdx').sort_values('reviewerIdx')
    encoded = pd.get_dummies(df_sorted[['authorGender']], columns=['authorGender'], prefix='gender')
    encoded = encoded.astype(int)
    for i in range(1, 23):
        encoded[f'random_col_{i}'] = np.random.uniform(-0.5, 0.5, len(encoded))
    return encoded.values
