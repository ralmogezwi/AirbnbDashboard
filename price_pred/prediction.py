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
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe



def prepare_for_model(X, ft_red_method=None):
        
    # Identify string columns
    string_columns = X.select_dtypes(include=['object']).columns

    # Get dummy variables for all string columns
    X_dummies = pd.get_dummies(X, columns=string_columns, drop_first=True, dtype=int)

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



def feature_reduction(X, method):
    
    # if method=='corr':
    #     # Step 1: Calculate the correlation between each feature in X and y
    #     correlations = X.corrwith(y).abs()  # Compute the absolute correlation of each feature with y

    #     # Step 2: Sort by correlation and select the top N features
    #     top_n = 100  # Number of features to select
    #     top_features = correlations.sort_values(ascending=False).head(top_n)

    #     # Step 3: Select the corresponding columns from X
    #     selected_columns = top_features.index

    #     # Filter X to include only the selected columns
    #     X_reduced = X[selected_columns]

    #     # Optionally, display the top correlated features
    #     print(top_features)

    
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

    # # Split the data into training and validation sets
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    
    # # Define hyperparameter space with probability distributions
    # space = {
    #     'gamma': hp.uniform('gamma', 0.001, 0.004),
    #     'reg_alpha': hp.uniform('reg_alpha', 0.5, 2),
    #     'reg_lambda': hp.uniform('reg_lambda', 0.1, 0.5),
    #     'max_depth': hp.quniform("max_depth", 10, 20, 1),
    #     'n_estimators': hp.quniform('n_estimators', 400, 550, 1),
    #     'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 0.4),
    #     'min_child_weight': hp.uniform('min_child_weight', 0.1, 0.4),
    #     'seed': 0, 'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
    #     'early_stopping_rounds': hp.quniform('early_stopping_rounds', 5, 20, 1)
    # }

    # def objective(space):
    #     # Set up XGBoost Algorithm
    #     xgb_reg = xgb.XGBRegressor(n_estimators=int(space['n_estimators']),
    #                                max_depth=int(space['max_depth']), gamma=space['gamma'],
    #                                reg_alpha=space['reg_alpha'], min_child_weight=space['min_child_weight'],
    #                                colsample_bytree=space['colsample_bytree'], reg_lambda=space['reg_lambda'],
    #                                early_stopping_rounds=int(space['early_stopping_rounds']), n_jobs=-1,
    #                                learning_rate=space['learning_rate'])

    #     # Define evaluation dataset
    #     evaluation = [(X_train, y_train), (X_valid, y_valid)]

    #     # Fit Algorithm to training data
    #     xgb_reg.fit(X_train, y_train, eval_set=evaluation, verbose=False)

    #     # Make predictions with validation data for hyperparameter tuning
    #     pred = xgb_reg.predict(X_valid)
    #     rmse = mean_squared_error(y_valid, pred, squared=False)
    #     return {'loss': rmse, 'status': STATUS_OK}

    # # Start Bayesian Hyperparameter Optimization with TPE Algorithm
    # trials = Trials()
    # hyperparams = fmin(fn=objective, space=space,
    #                    algo=tpe.suggest, max_evals=300, trials=trials)

    # # Retrieve best performing hyperparameters
    # space = {
    #     'max_depth': hyperparams['max_depth'],
    #     'reg_alpha': hyperparams['reg_alpha'],
    #     'gamma': hyperparams['gamma'], 'seed': 0,
    #     'reg_lambda': hyperparams['reg_lambda'],
    #     'n_estimators': hyperparams['n_estimators'],
    #     'learning_rate': hyperparams['learning_rate'],
    #     'colsample_bytree': hyperparams['colsample_bytree'],
    #     'min_child_weight': hyperparams['min_child_weight'],
    #     'early_stopping_rounds': hyperparams['early_stopping_rounds'],
    # }

    # # Set up final XGBoost Algorithm
    # model = xgb.XGBRegressor(n_estimators=int(space['n_estimators']),
    #                            max_depth=int(space['max_depth']), gamma=space['gamma'],
    #                            reg_alpha=space['reg_alpha'], min_child_weight=space['min_child_weight'],
    #                            colsample_bytree=space['colsample_bytree'], reg_lambda=space['reg_lambda'],
    #                            early_stopping_rounds=int(space['early_stopping_rounds']), n_jobs=-1,
    #                            learning_rate=space['learning_rate'], tree_method='gpu_hist')

    # # Define evaluation dataset
    # evaluation = [(X_train, y_train), (X_test, y_test)]

    # # Fit Algorithm to training data
    # model.fit(X_train, y_train, eval_set=evaluation, verbose=False)

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


# for city_code in ['nyc','la','chi']:
#     data = pd.read_csv(f'price_pred/{city_code}_clean.csv')
#     X, y = data.drop(columns=['price']), data[['price']]
#     # print(y)
#     X = prepare_for_model(X, None)
#     # print(X.columns)
#     model, y_pred = run_xgboost_regression(X, y)
#     print(y_pred)

#     with open(f'price_pred/{city_code}_model.pkl', 'wb') as f:
#         pickle.dump(model, f)