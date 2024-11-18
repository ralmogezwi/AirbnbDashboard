import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

def prepare_for_model(data, scaler=None):
    # Remove outliers if the 'price' column exists
    if 'price' in data.columns:
        upper_limit = data['price'].quantile(0.99)
        data = data[data['price'] < upper_limit]

    # List of explicitly specified columns
    categorical_columns = [
        'description', 'host_response_time', 'host_is_superhost',
        'host_has_profile_pic', 'host_identity_verified', 'room_type',
        'has_availability', 'instant_bookable'
    ]
    columns_from_20_onwards = data.columns[27:]
    categorical_columns = categorical_columns + list(columns_from_20_onwards)

    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Scale numeric features
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if scaler:
        data[numeric_columns] = scaler.transform(data[numeric_columns])  # Use provided fitted scaler
    else:
        scaler = MinMaxScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Handle training vs. prediction input
    if 'price' in data.columns:
        y = data['price']
        X = data.drop(columns=['price'])
        return X, y, scaler  # Return the scaler when fitting
    else:
        return data

def run_xgboost_regression(X, y, test_size=0.2, random_state=42):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize the XGBoost regressor with hyperparameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # Regression objective
        learning_rate=0.1,            # Step size shrinkage
        n_estimators=500,             # Number of boosting rounds
        max_depth=6,                  # Maximum depth of a tree
        subsample=0.8,                # Subsample ratio of the training set
        colsample_bytree=0.8,         # Subsample ratio of columns when constructing each tree
        random_state=random_state
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))  
    rmse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred), squared=False)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Return the model and predictions
    return model, y_pred

# # Process data and train model for each city
# for city_code in ['nyc', 'la', 'chi']:
#     print(f"Processing city: {city_code}")
#     data = pd.read_csv(f'price_pred/{city_code}_clean2.csv')
    
#     # Prepare data for model
#     X, y, scaler = prepare_for_model(data)
    
#     # Train the model and get predictions
#     model, y_pred = run_xgboost_regression(X, y)
    
#     # Save the trained model as a .pkl file
#     with open(f'price_pred/{city_code}_model2.pkl', 'wb') as f:
#         pickle.dump(model, f)

#     # Save the fitted scaler as a .pkl file
#     with open(f'price_pred/{city_code}_scaler.pkl', 'wb') as f:
#         pickle.dump(scaler, f)
