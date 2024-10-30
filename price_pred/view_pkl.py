import pickle
import pandas as pd

with open('price_pred/xgb_reg.pkl', 'rb') as file:
    data = pickle.load(file)


df = pd.DataFrame(data)