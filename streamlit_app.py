import pandas as pd
import xgboost as xgb
import pickle
import streamlit as st
import numpy as np
from math import exp
from scipy import stats
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import base64


# Background image path
background_image_path = "airbnb.png"
with open(background_image_path, "rb") as image_file:
    encoded_bg_image = base64.b64encode(image_file.read()).decode()

# CSS for styling
page_bg_img = f'''
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{encoded_bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: white;
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.6);
    z-index: 0;
}}
.stApp > div:first-child {{
    position: relative;
    z-index: 1;
}}
/* Set all primary text elements to white */
h1, h2, h3, h4, h5, h6, p, label {{
    color: white !important;
}}
.stNumberInput label, .stSelectbox label {{
    font-size: 1.2rem !important;
    font-weight: bold !important;
    color: #DCDCDC !important;
}}
/* Button styling */
.stButton>button {{
    color: white !important;
    background-color: #32CD32 !important;
    border: none;
}}
/* Markdown styling */
.css-1v3fvcr p {{
    color: white !important;
}}
/* Styling for prediction card */
.prediction-card {{
    background: linear-gradient(135deg, #ff6f61, #f7b731);
    padding: 20px;
    margin: 20px 0;
    border-radius: 15px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.6);
    text-align: center;
    font-size: 1.8rem;
    color: white;
    animation: reveal 1.5s ease;
}}
.white-text {{
    color: white !important;
}}
/* Transparent background for tabs */
div[data-baseweb="tab"] {{
    background-color: rgba(0, 0, 0, 0) !important;
}}
/* Transparent background for active tab content */
.css-1d391kg {{
    background-color: rgba(0, 0, 0, 0.4) !important;
}}
@keyframes reveal {{
    0% {{ transform: scale(0); opacity: 0; }}
    100% {{ transform: scale(1); opacity: 1; }}
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# Set up header and brief description
with st.container():
    st.title('Airbnb Dashboard')
    #st.markdown('Provide data about your Airbnb listing and get predictions!')
    st.markdown('Select your desired tab:')
#Initilaizing 3 tabs
tab1, tab2, tab3 = st.tabs(["Predictions", "Data Viz", "Suggestions"])

with tab1:
    # Begin new section for listings features
    st.subheader('Listing characteristics')
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox('City',
                                ('Washington, D.C.', 'New York City', 'Los Angeles'))
        accommodates = st.slider('Maximum Capacity', 1, 16, 4)
        bathrooms = st.slider('Number of bathrooms', 1, 9, 2)
        room_type = st.selectbox('Room Type',
                                ('Private room', 'Entire apartment', 'Shared room', 'Hotel room'))
        instant = st.selectbox('Can the listing be instantly booked?',
                            ('No', 'Yes'))
    with col2:
        beds = st.slider('Number of beds', 1, 32, 2)
        bedrooms = st.slider('Number of bedrooms', 1, 24, 2)
        min_nights = st.slider('Minimum number of nights', 1, 20, 3)
        amenities = st.multiselect(
            'Select available amenities',
            ['TV', 'Wifi', 'Netflix', 'Swimming pool', 'Hot tub', 'Gym', 'Elevator',
            'Fridge', 'Heating', 'Air Conditioning', 'Hair dryer', 'BBQ', 'Oven',
            'Security cameras', 'Workspace', 'Coffee maker', 'Backyard',
            'Outdoor dining', 'Host greeting', 'Beachfront', 'Patio',
            'Luggage dropoff', 'Furniture'],
            ['TV', 'Wifi'])

    # Section for host info
    st.markdown('---')
    st.subheader('Host Information')

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Host gender', ('Female', 'Male', 'Other/Corporation'))
        pic = st.selectbox('Does your host have a profile picture?', ('Yes', 'No'))
        dec = st.selectbox('Did your host write a description about the listing?', ('Yes', 'No'))
        super_host = st.selectbox('Is your host a superhost?', ('No', 'Yes'))
    with col2:
        verified = st.selectbox('Is your host verified?', ('Yes', 'No'))
        availability = st.selectbox('Is the listing available?', ('Yes', 'No'))
        response = st.selectbox('Response rate', (
            'Within an hour', 'Within a few hours', 'Within a day', 'Within a few days'))
        no_review = st.selectbox('Did your host get any review?', ('Yes', 'No'))

    host_since = st.slider(
        'Number of days your host has been using Airbnb',
        1, 5000, 2000)
    
    st.markdown('---')
    st.subheader("Guests' feedback")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.slider('Location rating', 1.0, 5.0, 4.0, step=0.5)
        checkin = st.slider('Checkin rating', 1.0, 5.0, 3.0, step=0.5)
    with col2:
        clean = st.slider('Cleanliness rating', 1.0, 5.0, 3.0, step=0.5)
        communication = st.slider('Communication rating', 1.0, 5.0, 4.0, step=0.5)
    with col3:
        value = st.slider('Value rating', 1.0, 5.0, 3.5, step=0.5)
        accuracy = st.slider('Accuracy rating', 1.0, 5.0, 4.2, step=0.5)

    # Center model prediction button
    _, col2, _ = st.columns(3)
    with col2:
        run_preds = st.button('Run the model')
        if run_preds:    
            # Load AI model
            with open('price_pred/xgb_reg.pkl', 'rb') as f:
                xgb_model = pickle.load(f)
                
            # One-hot encoding amenities
            options = ['TV', 'Wifi', 'Netflix', 'Swimming pool', 'Hot tub', 'Gym', 'Elevator',
                    'Fridge', 'Heating', 'Air Conditioning', 'Hair dryer', 'BBQ', 'Oven',
                    'Security cameras', 'Workspace', 'Coffee maker', 'Backyard',
                    'Outdoor dining', 'Host greeting', 'Beachfront', 'Patio',
                    'Luggage dropoff', 'Furniture']

            amens = [1 if i in amenities else 0 for i in options]
            tv, wifi, netflix, pool = amens[0], amens[1], amens[2], amens[3]
            tub, gym, elevator, fridge = amens[4], amens[5], amens[6], amens[7]
            heat, air, hair, bbq = amens[8], amens[9], amens[10], amens[11]
            oven, cams, workspace, coffee = amens[12], amens[13], amens[14], amens[15]
            backyard, outdoor, greet, beach = amens[16], amens[17], amens[18], amens[19]
            patio, luggage, furniture = amens[20], amens[21], amens[22]

            # One-hot encoding binary features
            dec = 1 if dec == 'Yes' else 0
            super_host = 1 if super_host == 'Yes' else 0
            pic = 1 if pic == 'Yes' else 0
            verified = 1 if verified == 'Yes' else 0
            availability = 1 if availability == 'Yes' else 0
            instant = 1 if instant == 'Yes' else 0
            gender = 1 if gender == 'Yes' else 0
            no_review = 0 if no_review == 'Yes' else 1

            # Encode room_type feature
            rooms = {
                'Private room': 1,
                'Entire home/apt': 2,
                'Shared room': 3,
                'Hotel room': 4
            }
            room_type = rooms.get(room_type)

            # Encode response_time feature
            responses = {
                'Within an hour': 1,
                'Within a few hours': 2,
                'Within a day': 3,
                'Within a few days': 4
            }
            response = responses.get(response)

            # Set up feature matrix for predictions
            X_test = pd.DataFrame(data=np.column_stack((
                dec, 2049.8854, response, 76.6324, 
                71.2640, super_host, 69.1362, pic, verified,
                room_type, accommodates, bathrooms, 
                bedrooms, beds, min_nights, 601.9025,
                27.1608, 332469.8321, availability,
                9.7215, 40.2549, 179.5286, 36.1675,
                9.8272, 0.9183, 827.2542, 238.0988, 
                4.6789, accuracy, clean, checkin, 
                communication, location, value,
                instant, 18.3119, 14.5421, 3.3295,
                0.3858, 0, 114, 0, 1.1963, 1,
                -0.3559, -0.7283, 0.5242, 17, 
                tv, netflix, gym, elevator, fridge,
                heat, hair, air, tub, oven, bbq, cams, 
                workspace, coffee, backyard, outdoor, 
                greet, pool, beach, patio, luggage, furniture,
                gender, 0.9643, 0.9029, 0.9650, no_review)))
            
            # Define the current feature names
            current_feature_names = ['description', 'host_since', 'host_response_time', 'host_response_rate',
                'host_acceptance_rate', 'host_is_superhost', 'host_listings_count', 'host_has_profile_pic', 
                'host_identity_verified', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 
                'minimum_nights', 'maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 
                'has_availability', 'availability_30', 'availability_90', 'availability_365', 'number_of_reviews', 
                'number_of_reviews_ltm', 'number_of_reviews_l30d', 'first_review', 'last_review', 
                'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 
                'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 
                'review_scores_value', 'instant_bookable', 'calculated_host_listings_count', 
                'calculated_entire', 'calculated_private', 'calculated_shared', 'neighborhood', 
                'neighborhood_group', 'inactive', 'reviews_month', 'responds', 'geo_x', 'geo_y', 'geo_z', 
                'property', 'tv', 'netflix', 'gym', 'elevator', 'fridge', 'heating', 'hair_dryer', 
                'air_conditioning', 'hot_tub', 'oven', 'bbq', 'security cameras', 'workspace', 'coffee', 
                'backyard', 'outdoor_dining', 'greets', 'pool', 'beachfront', 'patio', 'luggage', 'furniture', 
                'nlp_gender', 'sent_median', 'sent_mean', 'sent_mode', 'no_review']

            # Desired feature order
            desired_order = ['neighborhood_group', 'number_of_reviews', 'minimum_nights', 'beds', 'calculated_entire',
                'host_response_rate', 'number_of_reviews_l30d', 'host_is_superhost', 'heating', 'bedrooms', 
                'review_scores_accuracy', 'property', 'review_scores_communication', 'host_response_time', 'geo_x', 
                'number_of_reviews_ltm', 'room_type', 'host_has_profile_pic', 'tv', 'fridge', 'outdoor_dining', 
                'sent_mode', 'geo_z', 'coffee', 'bathrooms', 'availability_90', 'security cameras', 'inactive', 
                'maximum_nights_avg_ntm', 'last_review', 'minimum_nights_avg_ntm', 'greets', 'sent_median', 'responds', 
                'host_listings_count', 'patio', 'availability_30', 'calculated_shared', 'review_scores_rating', 
                'first_review', 'review_scores_location', 'reviews_month', 'review_scores_checkin', 'instant_bookable', 
                'host_since', 'hot_tub', 'bbq', 'netflix', 'workspace', 'description', 'pool', 'beachfront', 
                'luggage', 'nlp_gender', 'no_review', 'geo_y', 'backyard', 'review_scores_value', 'elevator', 
                'accommodates', 'review_scores_cleanliness', 'air_conditioning', 'availability_365', 'sent_mean', 
                'host_identity_verified', 'host_acceptance_rate', 'calculated_host_listings_count', 'neighborhood', 
                'hair_dryer', 'gym', 'oven', 'furniture', 'calculated_private', 'maximum_nights', 'has_availability']

            # Create the test input array (example data)
            data = np.column_stack((
                dec, 2049.8854, response, 76.6324, 71.2640, super_host, 69.1362, pic, verified, room_type, 
                accommodates, bathrooms, bedrooms, beds, min_nights, 601.9025, 27.1608, 332469.8321, 
                availability, 9.7215, 40.2549, 179.5286, 36.1675, 9.8272, 0.9183, 827.2542, 238.0988, 4.6789, 
                accuracy, clean, checkin, communication, location, value, instant, 18.3119, 14.5421, 3.3295, 
                0.3858, 0, 114, 0, 1.1963, 1, -0.3559, -0.7283, 0.5242, 17, tv, netflix, gym, elevator, 
                fridge, heat, hair, air, tub, oven, bbq, cams, workspace, coffee, backyard, outdoor, greet, 
                pool, beach, patio, luggage, furniture, gender, 0.9643, 0.9029, 0.9650, no_review))
            
            

            # Create DataFrame
            X_test = pd.DataFrame(data=data, columns=current_feature_names)

            # Reorder the DataFrame columns to match the desired order
            X_test_reordered = X_test[desired_order]
            predicted_price = exp(xgb_model.predict(X_test))
            X_test_reordered.loc[:, 'predicted_price'] = predicted_price
            
            st.info(f"Predicted price is ${round(predicted_price, 2)}")
            st.session_state['X_test_reordered'] = X_test_reordered


# Generate or load a large sample dataset
def generate_large_sample_data(n_samples=200):
    data = {
    'description': np.random.choice([0, 1], n_samples),
    'host_since': np.random.randint(500, 5000, n_samples),
    'host_response_time': np.random.choice([1, 2, 3, 4], n_samples),
    'host_response_rate': np.random.uniform(50, 100, n_samples),
    'host_acceptance_rate': np.random.uniform(50, 100, n_samples),
    'host_is_superhost': np.random.choice([0, 1], n_samples),
    'host_listings_count': np.random.randint(1, 20, n_samples),
    'host_has_profile_pic': np.random.choice([0, 1], n_samples),
    'host_identity_verified': np.random.choice([0, 1], n_samples),
    'room_type': np.random.choice([1, 2, 3, 4], n_samples),
    'accommodates': np.random.randint(1, 16, n_samples),
    'bathrooms': np.random.randint(1, 5, n_samples),
    'bedrooms': np.random.randint(1, 5, n_samples),
    'beds': np.random.randint(1, 10, n_samples),
    'minimum_nights': np.random.randint(1, 15, n_samples),
    'maximum_nights': np.random.randint(30, 365, n_samples),
    'minimum_nights_avg_ntm': np.random.uniform(1, 15, n_samples),
    'maximum_nights_avg_ntm': np.random.uniform(30, 365, n_samples),
    'has_availability': np.random.choice([0, 1], n_samples),
    'availability_30': np.random.randint(0, 30, n_samples),
    'availability_90': np.random.randint(0, 90, n_samples),
    'availability_365': np.random.randint(0, 365, n_samples),
    'number_of_reviews': np.random.randint(0, 500, n_samples),
    'number_of_reviews_ltm': np.random.randint(0, 200, n_samples),
    'number_of_reviews_l30d': np.random.randint(0, 30, n_samples),
    'first_review': np.random.uniform(1, 2000, n_samples),
    'last_review': np.random.uniform(1, 2000, n_samples),
    'review_scores_rating': np.random.uniform(1, 5, n_samples),
    'review_scores_accuracy': np.random.uniform(1, 5, n_samples),
    'review_scores_cleanliness': np.random.uniform(1, 5, n_samples),
    'review_scores_checkin': np.random.uniform(1, 5, n_samples),
    'review_scores_communication': np.random.uniform(1, 5, n_samples),
    'review_scores_location': np.random.uniform(1, 5, n_samples),
    'review_scores_value': np.random.uniform(1, 5, n_samples),
    'instant_bookable': np.random.choice([0, 1], n_samples),
    'calculated_host_listings_count': np.random.uniform(1, 50, n_samples),
    'calculated_entire': np.random.choice([0, 1], n_samples),
    'calculated_private': np.random.choice([0, 1], n_samples),
    'calculated_shared': np.random.choice([0, 1], n_samples),
    'neighborhood': np.random.randint(1, 20, n_samples),
    'neighborhood_group': np.random.randint(1, 5, n_samples),
    'inactive': np.random.choice([0, 1], n_samples),
    'reviews_month': np.random.uniform(0, 5, n_samples),
    'responds': np.random.choice([0, 1], n_samples),
    'geo_x': np.random.uniform(-180, 180, n_samples),
    'geo_y': np.random.uniform(-90, 90, n_samples),
    'geo_z': np.random.uniform(-90, 90, n_samples),
    'property': np.random.choice([0, 1], n_samples),
    'tv': np.random.choice([0, 1], n_samples),
    'netflix': np.random.choice([0, 1], n_samples),
    'gym': np.random.choice([0, 1], n_samples),
    'elevator': np.random.choice([0, 1], n_samples),
    'fridge': np.random.choice([0, 1], n_samples),
    'heating': np.random.choice([0, 1], n_samples),
    'hair_dryer': np.random.choice([0, 1], n_samples),
    'air_conditioning': np.random.choice([0, 1], n_samples),
    'hot_tub': np.random.choice([0, 1], n_samples),
    'oven': np.random.choice([0, 1], n_samples),
    'bbq': np.random.choice([0, 1], n_samples),
    'security cameras': np.random.choice([0, 1], n_samples),
    'workspace': np.random.choice([0, 1], n_samples),
    'coffee': np.random.choice([0, 1], n_samples),
    'backyard': np.random.choice([0, 1], n_samples),
    'outdoor_dining': np.random.choice([0, 1], n_samples),
    'greets': np.random.choice([0, 1], n_samples),
    'pool': np.random.choice([0, 1], n_samples),
    'beachfront': np.random.choice([0, 1], n_samples),
    'patio': np.random.choice([0, 1], n_samples),
    'luggage': np.random.choice([0, 1], n_samples),
    'furniture': np.random.choice([0, 1], n_samples),
    'nlp_gender': np.random.choice([0, 1], n_samples),
    'sent_median': np.random.uniform(0, 1, n_samples),
    'sent_mean': np.random.uniform(0, 1, n_samples),
    'sent_mode': np.random.uniform(0, 1, n_samples),
    'no_review': np.random.choice([0, 1], n_samples)
}
    # Convert to DataFrame and return
    return pd.DataFrame(data)

large_sample_data = generate_large_sample_data()
with open('price_pred/xgb_reg.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Predict and add predicted prices
large_sample_data['predicted_price'] = np.exp(xgb_model.predict(large_sample_data))

# Save in session state
st.session_state['X_test_reordered'] = large_sample_data

relevant_features = [
    'host_since', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 'availability_30',
    'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness',
    'review_scores_location', 'instant_bookable', 'tv', 'air_conditioning',
    'heating', 'pool'
]



# Correlation Visualization in Tab 2
with tab2:
    st.subheader("Feature Correlation with Predicted Price")

    # Check if prediction data is available
    if 'X_test_reordered' in st.session_state:
        X_test_reordered = st.session_state['X_test_reordered']

        # Filter to include only relevant features and the predicted price
        filtered_data = X_test_reordered[relevant_features + ['predicted_price']]
        correlations = filtered_data.corr()['predicted_price'].sort_values(ascending=False)
        
        # Select top features with highest correlation
        num_features = st.slider("Select number of top correlated features", 5, len(correlations)-1, 10)
        top_features = correlations.iloc[1:num_features+1]  # Exclude 'predicted_price' itself

        # Create an interactive Plotly bar plot
        fig = go.Figure(go.Bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            marker=dict(color=top_features.values, colorscale='Viridis'),
            text=top_features.values,
            textposition='auto'
        ))

        fig.update_layout(
            title="Top Correlated Features with Predicted Price",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Features",
            yaxis=dict(autorange="reversed"),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

    if 'X_test_reordered' in st.session_state:
        X_test_reordered = st.session_state['X_test_reordered']
        st.markdown('---')
        st.subheader("Price vs. Key Feature Plot")
        

        feature = st.selectbox(
        "Select a feature to compare with predicted price:",
        options=[col for col in filtered_data.columns if col != "predicted_price"]
    )

        # Plot with Plotly
        fig = px.scatter(filtered_data, x=feature, y="predicted_price", trendline="ols",
                        title=f"Predicted Price vs {feature.capitalize()}",
                        labels={"predicted_price": "Predicted Price", feature: feature.capitalize()})

        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)



    if 'disabled' not in st.session_state:
        st.session_state['disabled'] = False

        
    def disable():
        st.session_state['disabled'] = True



    # st.markdown('---')
    # st.subheader('About')
    # st.markdown('This a Data Science project unaffiliated with Airbnb')
    # st.markdown('Note that the predicted price is the amount hosts charge **per night**!')
    # st.markdown('Prediction accuracy is limited to listings in **Los Angeles** from **summer 2022**')
    # st.markdown('Sentiment Analysis prediction is restricted to one request due to limited compute resources')
    # transformer = 'https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest'
    # st.markdown('The deployed NLP model is the transformer [RoBERTa](%s)' % transformer)
    # thesis = 'https://github.com/jose-jaen/Airbnb'
    # st.markdown('Feel free to check the entirety of my Bachelor Thesis [here](%s)' % thesis)
    # linkedin = 'https://www.linkedin.com/in/jose-jaen/'
    # st.markdown('Reach out to [José Jaén Delgado](%s) for any questions' % linkedin)
