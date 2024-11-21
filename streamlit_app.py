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
from price_pred.prediction import prepare_for_model
import random




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
    st.title('Airbnb Pricing Dashboard')
    #st.markdown('Provide data about your Airbnb listing and get predictions!')
    st.markdown('Select your desired tab:')
#Initilaizing 3 tabs
tab0, tab1, tab2, tab3 = st.tabs(["About", "Price Generation", "Data Vis", "Update Suggestions"])

with tab0:
    st.subheader('Welcome to the Airbnb Pricing Dashboard!')
    st.markdown('Are you an Airbnb host? Are you interested in becoming one? This dashboard is for you!')
    st.markdown('Airbnb hosts face a crucial challenge when putting their property up for rental: What should you charge guests per night?')
    st.markdown('There are many factors that influence the market price of an Airbnb listing, far too many to consider comprehensively. That is why we have created a dashboard which allows you to enter information about your listing and provides you with an appropriate price that will generate bookings and revenue. Our price generation model uses an Extreme Gradient Boosting algorithm to consider all the important factors and find you the right price for your listing.')
    st.markdown('In addition to Price Generation, the dashboard features two additional tabs: Data Vis and Update Suggestions.')
    st.markdown('Data Vis allows you to see which listing variables correlate most highly with price in your city.')
    st.markdown('Update Suggestions allows you to set a price you would like to charge for your listing and provides an itemized list of actions you should take to match your listing profile with that price point. For example, if you are dissatisfied with your generated price of $100 and would like to charge $200, we may suggest that you increase the number of people you can accomodate or your average guest rating.')


with tab1:
    # Begin new section for listings features
    st.subheader('Listing characteristics')
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox('City',
                                ('Chicago', 'New York City', 'Los Angeles'))
        
        if city=='Chicago':
            city_code = 'chi'
        if city=='New York City':
            city_code = 'nyc'
        if city=='Los Angeles':
            city_code = 'la'

        data = pd.read_csv(f'price_pred/{city_code}_clean.csv')

        # neighborhood = st.selectbox('Neighborhood',
        #                         data['neighbourhood_cleansed'].unique())
        # neighborhood_group = st.selectbox('Neighborhood Group',
        #                         data['neighbourhood_group_cleansed'].unique())
        accommodates = st.slider('Maximum Capacity', int(data['accommodates'].min()), int(data['accommodates'].max()), int(data['accommodates'].median()))
        bathrooms = st.slider('Number of bathrooms', float(data['bathrooms'].min()), float(data['bathrooms'].max()), float(data['bathrooms'].median()), step=.5)
        # bathroom_type = st.selectbox('Bathroom Type',
        #                         data['bathroom_type'].unique())
        room_type = st.selectbox('Room Type',
                                data['room_type'].unique())
        instant = st.selectbox('Can the listing be instantly booked?',
                            ('Yes', 'No'))
    with col2:
        beds = st.slider('Number of beds', int(data['beds'].min()), int(data['beds'].max()), int(data['beds'].median()))
        bedrooms = st.slider('Number of bedrooms', int(data['bedrooms'].min()), int(data['bedrooms'].max()), int(data['bedrooms'].median()))
        min_nights = st.slider('Minimum number of nights', int(data['minimum_nights'].min()), int(data['minimum_nights'].max()), int(data['minimum_nights'].median()))
        max_nights = st.slider('Maximum number of nights', int(data['maximum_nights'].min()), int(data['maximum_nights'].max()), int(data['minimum_nights'].median()))
        
        amen_options = list(data.columns[-20:])
        amenities = st.multiselect(
            'Select available amenities',
            amen_options,
            # random.sample(amen_options, 5)
            )

    # Section for host info
    st.markdown('---')
    st.subheader('Host Information')

    col1, col2 = st.columns(2)
    with col1:
        pic = st.selectbox('Does your host have a profile picture?', ('Yes', 'No'))
        dec = st.selectbox('Did your host write a description about the listing?', ('Yes', 'No'))
        super_host = st.selectbox('Is your host a superhost?', ('No', 'Yes'))
        response_rate = st.slider('Response rate', 0.0, 1.0, .8, step=.1)
        accept_rate = st.slider('Acceptance rate', 0.0, 1.0, .8, step=.1)
    with col2:
        verified = st.selectbox('Is your host verified?', ('Yes', 'No'))
        availability = st.selectbox('Is the listing available?', ('Yes', 'No'))
        response_time = st.selectbox('Response time', data['host_response_time'].unique())
        num_review = st.slider('Number of reviews', int(data['number_of_reviews'].min()), int(data['number_of_reviews'].max()), int(data['number_of_reviews'].median()))
        num_listings = st.slider('Number of listings', int(data['host_listings_count'].min()), int(data['host_listings_count'].max()), int(data['host_listings_count'].median()))
    
    st.markdown('---')
    st.subheader("Guests' feedback")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        overall = st.slider('Overall rating', 1.0, 5.0, 3.0, step=0.1)
        location = st.slider('Location rating', 1.0, 5.0, 3.0, step=0.1)
        checkin = st.slider('Checkin rating', 1.0, 5.0, 3.0, step=0.1)
    with col2:
        clean = st.slider('Cleanliness rating', 1.0, 5.0, 3.0, step=0.1)
        communication = st.slider('Communication rating', 1.0, 5.0, 3.0, step=0.1)
    with col3:
        value = st.slider('Value rating', 1.0, 5.0, 3.0, step=0.1)
        accuracy = st.slider('Accuracy rating', 1.0, 5.0, 3.0, step=0.1)

    # Center model prediction button
    _, col2, _ = st.columns(3)
    with col2:
        run_preds = st.button('Run the model')
        if run_preds:    
            # Load AI model
            with open(f'price_pred/{city_code}_model.pkl', 'rb') as f:
                xgb_model = pickle.load(f)

            amens = [1 if i in amenities else 0 for i in amen_options]

            # One-hot encoding binary features
            dec = 1 if dec == 'Yes' else 0
            super_host = 1 if super_host == 'Yes' else 0
            pic = 1 if pic == 'Yes' else 0
            verified = 1 if verified == 'Yes' else 0
            availability = 1 if availability == 'Yes' else 0
            instant = 1 if instant == 'Yes' else 0

            X_cols = data.columns[data.columns != "price"]
            # st.info(len(amens))

            X_test = pd.DataFrame(columns=X_cols, data=[[
                        dec,
                        response_time,
                        response_rate,
                        accept_rate,
                        super_host,
                        num_listings,
                        pic,
                        verified,
                        # neighborhood,
                        # neighborhood_group,
                        room_type,
                        accommodates,
                        bathrooms,
                        bedrooms,
                        beds,
                        min_nights,
                        max_nights,
                        availability,
                        num_review,
                        overall,
                        accuracy,
                        clean,
                        checkin,
                        communication,
                        location,
                        value,
                        instant,
                        # bathroom_type,
                        ] + amens])
            
            # Prepare for model
            X_test = prepare_for_model(X_test)

            # Ensure X_test has exactly the same columns as the model's feature names
            expected_features = xgb_model.get_booster().feature_names  # Extract expected features from model

            # Align columns in X_test with the expected features
            X_test = X_test.reindex(columns=expected_features)

            # Get predicted price
            st.info(f"Predicted price is ${round(abs(float(xgb_model.predict(X_test)[0])), 2)}")


# Correlation Visualization in Tab 2
with tab2:
    vis_city = st.selectbox('City',
                            ('Chicago', 'New York City', 'Los Angeles'), key='vis')
    
    if vis_city=='Chicago':
        vis_city_code = 'chi'
    if vis_city=='New York City':
        vis_city_code = 'nyc'
    if vis_city=='Los Angeles':
        vis_city_code = 'la'

    vis_data = pd.read_csv(f'price_pred/{vis_city_code}_clean.csv')
    
    # Save in session state
    st.session_state['vis_data'] = vis_data
    
    st.subheader("Feature Correlation with Price")

    # Check if prediction data is available
    if 'vis_data' in st.session_state:
        
        vis_data = st.session_state['vis_data']
        data_quant = vis_data.drop(columns=['room_type', 'host_response_time'])
        from scipy.stats import zscore

        # Calculate the Z-score for the 'price' column
        data_quant['z_score'] = zscore(data_quant['price'])

        # Filter the DataFrame to remove rows where the Z-score is greater than 3 or less than -3
        data_filtered = data_quant[(data_quant['z_score'].abs() <= 3)]

        # Optionally, drop the Z-score column after filtering
        data_quant_filtered = data_filtered.drop(columns=['z_score'])

        # # Filter to include only relevant features and the predicted price
        # filtered_data = X_test_reordered[relevant_features + ['predicted_price']]
        correlations = data_quant_filtered.corr()['price'].sort_values(ascending=False)
        
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
            title="Top Correlated Features with Price",
            xaxis_title="Correlation Coefficient",
            yaxis_title="Features",
            yaxis=dict(autorange="reversed"),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('---')
        st.subheader("Price vs. Key Feature Plot")
        

        feature = st.selectbox(
        "Select a feature to compare with predicted price:",
        options=[col for col in data_quant_filtered.columns if col != "price"]
    )

        # Plot with Plotly
        fig = px.scatter(data_quant_filtered, x=feature, y="price", trendline="ols",
                        title=f"Price vs {feature.capitalize()}",
                        labels={"price": "Price", feature: feature.capitalize()})

        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

with tab3:
    st.header("Suggestions for Improving Your Listing")
    # Collect listing details from the user
    st.subheader("Input Your Listing Details")
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox('City',
                            ('Chicago', 'New York City', 'Los Angeles'),
                            key="city_selector")

        if city == 'Chicago':
            city_code = 'chi'
        if city == 'New York City':
            city_code = 'nyc'
        if city == 'Los Angeles':
            city_code = 'la'

        data = pd.read_csv(f'price_pred/{city_code}_clean.csv')

        accommodates = st.slider('Maximum Capacity', 
                                  int(data['accommodates'].min()), 
                                  int(data['accommodates'].max()), 
                                  int(data['accommodates'].median()),
                                  key="accommodates_slider")
        bathrooms = st.slider('Number of bathrooms', 
                               float(data['bathrooms'].min()), 
                               float(data['bathrooms'].max()), 
                               float(data['bathrooms'].median()), 
                               step=.5,
                               key="bathrooms_slider")
        room_type = st.selectbox('Room Type',
                                 data['room_type'].unique(),
                                 key="room_type_selector")
        instant = st.selectbox('Can the listing be instantly booked?',
                                ('Yes', 'No'),
                                key="instant_booking_selector")
    with col2:
        beds = st.slider('Number of beds', 
                         int(data['beds'].min()), 
                         int(data['beds'].max()), 
                         int(data['beds'].median()),
                         key="beds_slider")
        bedrooms = st.slider('Number of bedrooms', 
                              int(data['bedrooms'].min()), 
                              int(data['bedrooms'].max()), 
                              int(data['bedrooms'].median()),
                              key="bedrooms_slider")
        min_nights = st.slider('Minimum number of nights', 
                               int(data['minimum_nights'].min()), 
                               int(data['minimum_nights'].max()), 
                               int(data['minimum_nights'].median()),
                               key="min_nights_slider")
        max_nights = st.slider('Maximum number of nights', 
                               int(data['maximum_nights'].min()), 
                               int(data['maximum_nights'].max()), 
                               int(data['minimum_nights'].median()),
                               key="max_nights_slider")

        amen_options = list(data.columns[-20:])
        amenities = st.multiselect('Select available amenities',
                                   amen_options,
                                   key="amenities_multiselect")

    # Section for host info
    st.markdown('---')
    st.subheader('Host Information')

    col1, col2 = st.columns(2)
    with col1:
        pic = st.selectbox('Does your host have a profile picture?', 
                           ('Yes', 'No'),
                           key="profile_pic_selector")
        dec = st.selectbox('Did your host write a description about the listing?', 
                           ('Yes', 'No'),
                           key="description_selector")
        super_host = st.selectbox('Is your host a superhost?', 
                                  ('No', 'Yes'),
                                  key="super_host_selector")
        response_rate = st.slider('Response rate', 
                                  0.0, 1.0, .8, step=.1,
                                  key="response_rate_slider")
        accept_rate = st.slider('Acceptance rate', 
                                0.0, 1.0, .8, step=.1,
                                key="accept_rate_slider")
    with col2:
        verified = st.selectbox('Is your host verified?', 
                                ('Yes', 'No'),
                                key="verified_selector")
        availability = st.selectbox('Is the listing available?', 
                                    ('Yes', 'No'),
                                    key="availability_selector")
        response_time = st.selectbox('Response time', 
                                     data['host_response_time'].unique(),
                                     key="response_time_selector")
        num_review = st.slider('Number of reviews', 
                               int(data['number_of_reviews'].min()), 
                               int(data['number_of_reviews'].max()), 
                               int(data['number_of_reviews'].median()),
                               key="num_review_slider")
        num_listings = st.slider('Number of listings', 
                                 int(data['host_listings_count'].min()), 
                                 int(data['host_listings_count'].max()), 
                                 int(data['host_listings_count'].median()),
                                 key="num_listings_slider")

    st.markdown('---')
    st.subheader("Guests' feedback")

    col1, col2, col3 = st.columns(3)
    with col1:
        overall = st.slider('Overall rating', 
                            1.0, 5.0, 3.0, step=0.1,
                            key="overall_rating_slider")
        location = st.slider('Location rating', 
                             1.0, 5.0, 3.0, step=0.1,
                             key="location_rating_slider")
        checkin = st.slider('Checkin rating', 
                            1.0, 5.0, 3.0, step=0.1,
                            key="checkin_rating_slider")
    with col2:
        clean = st.slider('Cleanliness rating', 
                          1.0, 5.0, 3.0, step=0.1,
                          key="cleanliness_rating_slider")
        communication = st.slider('Communication rating', 
                                  1.0, 5.0, 3.0, step=0.1,
                                  key="communication_rating_slider")
    with col3:
        value = st.slider('Value rating', 
                          1.0, 5.0, 3.0, step=0.1,
                          key="value_rating_slider")
        accuracy = st.slider('Accuracy rating', 
                             1.0, 5.0, 3.0, step=0.1,
                             key="accuracy_rating_slider")
        
    desired_price = st.number_input('Enter Your Desired Price', 
                                        min_value=50, max_value=2000, value=200,
                                        key="desired_price_input")
       # Generate Suggestions
    st.markdown('---')

    if st.button("Get Suggestions"):

        # Sample rules for suggestions
        suggestions = []

        # Instant book suggestion
        if instant == 'No':
            suggestions.append("Enable instant booking to make your listing more accessible to guests.")

        # Price suggestion
        similar_accommodates_data = data[data['accommodates'] == accommodates]
        if not similar_accommodates_data.empty:
            avg_price = similar_accommodates_data['price'].mean()
            avg_price = round(avg_price)
            if desired_price > avg_price * 1.5:
                suggestions.append(f"Your desired price is significantly higher than the market average for listings with similar capacity (${avg_price}). Consider lowering it to stay competitive.")
            elif desired_price < avg_price * 0.7:
                suggestions.append(f"Your desired price is significantly lower than the market average for listings with similar capacity (${avg_price}). Consider raising it to avoid underpricing.")
            else:
                pass
            # Amenities suggestion

        if len(amenities) < 5:
            suggestions.append("Add more amenities like Wi-Fi, Kitchen, or Air Conditioning to make your listing more appealing.")

        # Superhost suggestion
        if super_host == 'No':
            suggestions.append("Strive to become a superhost by maintaining high ratings and response rates.")

        # Reviews suggestion
        if num_review < 10:
            suggestions.append("Encourage more guests to leave reviews to build trust and attract more bookings.")
        
            # Response time suggestion
        if response_time not in ['within an hour', 'within a few hours']:
            suggestions.append("Improve your response time to 'within an hour' or 'within a few hours' to enhance guest satisfaction.")

        # Response rate suggestion
        if response_rate < 0.8:
            suggestions.append("Increase your response rate to at least 80% to improve guest confidence in your responsiveness.")

        # Acceptance rate suggestion
        if accept_rate < 0.7:
            suggestions.append("Increase your acceptance rate to at least 70% to attract more bookings and improve your listing's ranking.")

        # Availability suggestion
        if availability == 'No':
            suggestions.append("Ensure your listing is marked as available to avoid losing potential bookings.")

        # Cleanliness rating suggestion
        if clean < 4.5:
            suggestions.append("Focus on improving cleanliness to achieve a rating above 4.5. Guests value cleanliness highly when choosing a property.")

        # Communication rating suggestion
        if communication < 4.5:
            suggestions.append("Enhance your communication with guests to achieve a rating above 4.5. Prompt and clear communication builds trust.")

        # Value rating suggestion
        if value < 4.5:
            suggestions.append("Ensure your pricing reflects the value of your property. Guests expect value for their money.")

        # Amenities quantity suggestion
        if len(amenities) < len(data.columns[27:]):
            suggestions.append("Expand the amenities offered to match similar listings in your area.")

        # Air conditioning suggestion
        if 'Air conditioning' not in amenities:
            suggestions.append("Add air conditioning to make your listing more comfortable for guests during hot seasons.")

        # Free parking suggestion
        if 'Free parking on premises' not in amenities:
            suggestions.append("Offer free parking on premises to attract guests traveling by car.")

        # Long-term stays suggestion
        if max_nights < 30:
            suggestions.append("Allow longer maximum stays to attract guests looking for extended stays.")

        # Pet-friendly suggestion
        if 'Pet-friendly' not in amenities:
            suggestions.append("Consider allowing pets to widen your potential guest base.")

        # Overall rating suggestion
        if overall < 4.5:
            suggestions.append("Work on improving your overall rating to above 4.5 to build credibility and attract more bookings.")

        # Location rating suggestion
        if location < 4.5:
            suggestions.append("Enhance the location experience for guests by providing local tips or improving accessibility.")

        # Safety suggestion
        if 'Smoke alarm' not in amenities or 'Carbon monoxide alarm' not in amenities:
            suggestions.append("Ensure safety by adding smoke alarms and carbon monoxide alarms to your property.")

        # Photos suggestion
        if pic == 'No':
            suggestions.append("Add high-quality photos of your property to make it more visually appealing to potential guests.")

        

        # Display suggestions
        if suggestions:
            st.write("### Suggestions for Improvement:")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")


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
