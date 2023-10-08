import pandas as pd
import numpy as np
import pickle

import plotly.graph_objects as go

from .mrmr import mrmr_selection
from .utils import LR, RF, XGB, KNN

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder



class PredictConversions:
    def __init__(self,
                 data: str=None,
                 model: str=None,
                 index_row: int=1, 
                 ) -> None:
        
        self.data = data
        self.model = model
        self.index_row = index_row

    def predict(self):
        df_predict = pd.read_csv(self.data, sep='|')

        encoder = LabelEncoder()

        # Encode the string data into a new column
        string_data = ['geo', 'os_type', 'os', 'os_version', 'device_type', 'device',
                       'browser', 'lang', 'proxy', 'netspeed']

        for i in string_data:
            df_predict[i] = encoder.fit_transform(df_predict[i])
        
        k_best_features = pickle.load(open('save_models/best_features.pickle', 'rb'))
        X_pred = df_predict.loc[int(self.index_row), k_best_features]

        if self.model == 'LR':
            model = pickle.load(open('save_models/lr_model.pickle', 'rb'))
        elif self.model == 'KNN':
            model = pickle.load(open('save_models/knn_model.pickle', 'rb'))
        elif self.model == 'RF':
            model = pickle.load(open('save_models/rf_model.pickle', 'rb'))
        elif self.model == 'XGB':
            model = pickle.load(open('save_models/xgb_model.pickle', 'rb'))
    

        y_pred_prob = model.predict_proba(np.array(X_pred).reshape(1, -1))

        return y_pred_prob[:,1]