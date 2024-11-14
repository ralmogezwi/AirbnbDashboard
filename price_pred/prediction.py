import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.impute import KNNImputer



def prepare_for_model(X, ft_red_method=None):
        
    # Identify string columns
    string_columns = X.select_dtypes(include=['object']).columns

    # Get dummy variables for all string columns
    X_dummies = pd.get_dummies(X, columns=string_columns, drop_first=True)

    # if X_dummies.shape[0]==1:
    #     return X_dummies

    # print(X_dummies.shape)
    # print(X_dummies.dtypes)
    
    # Replace NaNs with a specified value (e.g., column mean or zero)
    X_dummies.fillna(X_dummies.median(), inplace=True)  # Replace NaN with column median

    # # Initialize the KNN imputer
    # knn_imputer = KNNImputer(n_neighbors=5)

    # # Apply KNN imputation
    # X_dummies_imputed = knn_imputer.fit_transform(X_dummies)

    # # Convert the result back to a DataFrame
    # X_dummies_imputed_df = pd.DataFrame(X_dummies_imputed, columns=X_dummies.columns)

    # Feature reduction
    if ft_red_method:
        X_dummies = feature_reduction(X_dummies, ft_red_method)

    # return X_dummies_imputed_df
    return X_dummies



def feature_reduction(X, y, method='corr'):
    
    if method=='corr':
        # Step 1: Calculate the correlation between each feature in X and y
        correlations = X.corrwith(y).abs()  # Compute the absolute correlation of each feature with y

        # Step 2: Sort by correlation and select the top N features
        top_n = 100  # Number of features to select
        top_features = correlations.sort_values(ascending=False).head(top_n)

        # Step 3: Select the corresponding columns from X
        selected_columns = top_features.index

        # Filter X to include only the selected columns
        X_reduced = X[selected_columns]

        # Optionally, display the top correlated features
        print(top_features)

    
    if method=='pca':
        pca = PCA(n_components=10)
        X_reduced = pca.fit_transform(X)
    
    if method=='svd':
        
        # Assuming X_sparse is your sparse matrix
        X_sparse = csr_matrix(X)

        # Define the number of components you want to retain
        n_components = 50

        # Apply SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X_sparse)

    else:
        pass

    return X_reduced



def run_xgboost_regression(X, y, test_size=0.2, random_state=42):
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize the XGBoost regressor
    model = xgb.XGBRegressor()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Return the model and predictions
    return model, y_pred


for city_code in ['nyc','la','chi']:
    data = pd.read_csv(f'price_pred/{city_code}_clean.csv')
    X, y = data.drop(columns=['price']), data[['price']]
    X = prepare_for_model(X, None)
    # print(X.columns)
    model, y_pred = run_xgboost_regression(X, y)

    with open(f'price_pred/{city_code}_model.pkl', 'wb') as f:
        pickle.dump(model, f)