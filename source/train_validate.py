import pandas as pd
import numpy as np
import pickle

import plotly.graph_objects as go

from .mrmr import mrmr_selection
from .utils import LR, RF, XGB, KNN

from sklearn.metrics import classification_report


class TrainModel:
    def __init__(self, 
                 dataset_root_train: str=None, 
                 dataset_root_test: str=None, 
                 model: str=None, 
                 k_best_features: int=10,
                 n_calls: int=30,
                 model_treshold: int=0.5
                 ) -> None:
        
        self.dataset_root_train = dataset_root_train
        self.dataset_root_test = dataset_root_test
        self.model = model
        self.k_best_features = k_best_features
        self.n_calls = n_calls
        self.model_treshold = model_treshold

    @staticmethod
    def get_best_features(selected_features, score): 
        feature_scores = pd.DataFrame(score, index=selected_features, columns=['score'])
        
        y_ = np.arange(0, len(selected_features))
        fig = go.Figure(go.Bar(
            x=np.array(feature_scores).flatten(),
            y=feature_scores.index,
            orientation='h'))
        
        fig.update_layout(height=700, width=500)
        fig.update_layout(
            yaxis = dict(
                tickmode = 'linear',
                tick0 = 0.25,
                dtick = 0.25
            )
        )
        
        fig.show()

    def train_model(self):
        df_train = pd.read_csv(self.dataset_root_train)
        df_test = pd.read_csv(self.dataset_root_test)

        X_train = df_train.drop(['cp'], axis=1)
        y_train = df_train['cp']
    
        X_test = df_test.drop(['cp'], axis=1)
        y_test = df_test['cp']


        selected_features, score = mrmr_selection(X_train, y_train, K = self.k_best_features, relevance = 'mi_class', redundancy = 'pearson', denominator = 'mean')
        
        all_features, all_score = mrmr_selection(X_train, y_train, K = 22, relevance = 'mi_class', redundancy = 'pearson', denominator = 'mean')
        TrainModel.get_best_features(all_features, all_score)
        
        with open('save_models/best_features.pickle', 'wb') as file:
            pickle.dump(selected_features, file)

        X_train_best = X_train.loc[:, selected_features]
        X_test_best = X_test.loc[:, selected_features]

        if self.model == 'LR':
            model_train = LR(X_train_best, y_train, n_calls=self.n_calls, normalize=False)
            y_pred_prob = model_train.predict_proba(X_test_best)
        elif self.model == 'KNN':
            model_train = KNN(X_train_best, y_train, n_calls=self.n_calls, normalize=False)
            y_pred_prob = model_train.predict_proba(X_test_best)
        elif self.model == 'RF':
            model_train = RF(X_train_best, y_train, n_calls=self.n_calls, normalize=False)
            y_pred_prob = model_train.predict_proba(X_test_best)
        elif self.model == 'XGB':
            model_train = XGB(X_train_best, y_train, n_calls=self.n_calls, normalize=False)
            y_pred_prob = model_train.predict_proba(X_test_best)
        else:
            raise Exception('You must choose a model!')
        

        model_threshold = self.model_treshold
        y_pred_treshold = (y_pred_prob[:,1] >= model_threshold).astype(int)
        
        print('')
        print('Best k features:', selected_features)
        print('')
        print('Metrics:', classification_report(y_pred_treshold, y_test))
        

        return y_pred_prob[:,1]
