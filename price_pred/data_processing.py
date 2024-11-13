import pandas as pd
import json
import numpy as np


def clean_data(df):
    
    data = df[[
        # 'host_since',
        'description',
        # 'host_location',
        'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost',
        # 'host_neighbourhood',
        'host_listings_count',
        # 'host_total_listings_count',
        'host_has_profile_pic', 'host_identity_verified',
        'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
        # 'latitude', 'longitude',
        # 'property_type',
        'room_type', 'accommodates', 'bathrooms',
        'bathrooms_text', 'bedrooms', 'beds', 'amenities',
        'minimum_nights', 'maximum_nights',
        # 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights',
        # 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
        'has_availability',
        # 'availability_30', 'availability_60', 'availability_90', 'availability_365',
        'number_of_reviews',
        # 'number_of_reviews_ltm', 'number_of_reviews_l30d',
        # 'first_review', 'last_review',
        'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'instant_bookable',
        # 'calculated_host_listings_count',
        # 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
        # 'reviews_per_month',
        'price']]

    # Convert percentages to numbers
    for feature in ['host_response_rate', 'host_acceptance_rate']:
        data[feature] = data[feature].str.replace('%', '').astype(float) / 100
    
    # Convert t/f to integers
    for feature in ['host_has_profile_pic', 'host_is_superhost', 'host_identity_verified', 'has_availability', 'instant_bookable']:
        data[feature] = data[feature].map({'t': 1, 'f': 0})
    
    # See whether description exists
    data['description'] = np.where(data['description'].isna() | (data['description'] == ''), 0, 1)
    
    # Convert bathroom text to bathroom type
    data['bathroom_type'] = data['bathrooms_text'].apply(lambda data: 'private' if 'private' in str(data).lower() else ('shared' if 'shared' in str(data).lower() else None))
    data.drop(columns=['bathrooms_text'], inplace=True)

    # Convert the strings to lists
    data['amenities'] = data['amenities'].apply(json.loads)

    # Flatten the list and count occurrences
    all_amenities = [amenity for sublist in data['amenities'] for amenity in sublist]
    amenity_counts = pd.Series(all_amenities).value_counts()

    # Get the top 20 most common amenities
    top_20_amenities = amenity_counts.nlargest(20).index.tolist()

    # Create new columns for only the top 20 amenities
    for amenity in top_20_amenities:
        data[amenity] = data['amenities'].apply(lambda data: 1 if amenity in data else 0)

    data.drop(columns=['amenities'], inplace=True)

    # # Convert dates to floats
    # for feature in ['host_since', 'first_review', 'last_review']:   
    #     # Convert the 'date_column' to datetime
    #     data[feature] = pd.to_datetime(data[feature])

    #     # Define the reference date
    #     reference_date = pd.to_datetime('2007-01-01')

    #     # Calculate the number of days since the reference date
    #     data[feature] = (data[feature] - reference_date).dt.days
    
    # Convert prices from strings to floats
    data['price'] = data['price'].str.replace('[$,]', '', regex=True).astype(float)

    # Filter rows where 'price' is not NaN
    data = data[data['price'].notna()]
    
    return data



nyc_listings = pd.read_csv('data/nyc_listings.csv')
la_listings = pd.read_csv('data/la_listings.csv')
chi_listings = pd.read_csv('data/chi_listings.csv')

nyc_clean = clean_data(nyc_listings)
la_clean = clean_data(la_listings)
chi_clean = clean_data(chi_listings)

nyc_clean.to_csv('price_pred/nyc_clean.csv', index=False)
la_clean.to_csv('price_pred/la_clean.csv', index=False)
chi_clean.to_csv('price_pred/chi_clean.csv', index=False)

