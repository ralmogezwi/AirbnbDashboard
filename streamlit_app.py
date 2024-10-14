import streamlit as st

# Set up header and brief description
with st.container():
    st.title('Airbnb Price Predictor')
    st.markdown('Provide data about your Airbnb listing and get predictions!')

# Begin new section for listings features
st.markdown('---')
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