import pandas as pd
import json
import numpy as np

def clean_data(data):
    # Handle number_of_reviews = 0
    zero_review_mask = data['number_of_reviews'] == 0
    review_columns = [
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 
        'review_scores_value'
    ]
    data.loc[zero_review_mask, review_columns] = 0

    # Handle empty values in host_response_time, host_response_rate, host_acceptance_rate, host_is_superhost
    fill_lowest_columns = ['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost']
    for col in fill_lowest_columns:
        if data[col].dtype == 'object':
            data[col].fillna(data[col].value_counts().idxmin(), inplace=True)
        else:
            data[col].fillna(data[col].min(), inplace=True)

    return data


nyc_listings = pd.read_csv('price_pred/nyc_clean.csv')
la_listings = pd.read_csv('price_pred/la_clean.csv')
chi_listings = pd.read_csv('price_pred/chi_clean.csv')

nyc_clean = clean_data(nyc_listings)
la_clean = clean_data(la_listings)
chi_clean = clean_data(chi_listings)

nyc_clean.to_csv('price_pred/nyc_clean2.csv', index=False)
la_clean.to_csv('price_pred/la_clean2.csv', index=False)
chi_clean.to_csv('price_pred/chi_clean2.csv', index=False)