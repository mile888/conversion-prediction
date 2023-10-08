import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from skopt.space import Real, Categorical, Integer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from skopt import gbrt_minimize, gp_minimize

import pickle

import warnings
warnings.filterwarnings("ignore")

   

    
# Logistic Regression    
def LR(X, y, n_calls=20, normalize=False):
    """ Logistic Regression
    
        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_calls:    The number of GP function evaluations.
            normalize:  Scaling input data.
    """
    
    if normalize is True: 
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X) 
        
    class_weights = {0: 1.0, 1: 100.0}  
    class_weights = class_weights

    bayes_model = LogisticRegression(random_state=0)

    space  = [Categorical({'lbfgs', 'newton-cg', 'newton-cholesky', 'sag'}, name='solver'),
              Categorical({'none', 'l2'},   name='penalty'),
              Real(1e-5, 1000,  name='C'),
              ]
    
    @use_named_args(space)
    def objective(**params):
        bayes_model.set_params(**params)
        return -np.mean(cross_val_score(bayes_model, X, y, cv=10, n_jobs=-1, scoring="roc_auc"))
    
    res_bayes_model = gbrt_minimize(objective, space, n_calls=n_calls, random_state=0)
       
    
    model = LogisticRegression(solver=res_bayes_model.x[0],
                               penalty=res_bayes_model.x[1],
                               C=res_bayes_model.x[2], 
                               random_state=0,
                               ) 
    
    model.fit(X, y)
    
    with open('save_models/lr_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open('save_models/lr_param.pickle', 'wb') as f:
        pickle.dump(res_bayes_model.x, f)
    
    return model

        
       
        
# K-Near Neighbors
def KNN(X, y, n_calls=20, normalize=False):
    """ K-Neighbors for Classification
    
        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_calls:    The number of GP function evaluations.
            normalize:  Scaling input data.
    """

    if normalize is True: 
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X) 
 
    bayes_model = KNeighborsClassifier()

    space  = [Integer(3, 100,  name='n_neighbors'),
              Categorical({'uniform', 'distance'},   name='weights'),
             ]
    
    @use_named_args(space)
    def objective(**params):
        bayes_model.set_params(**params)
        return -np.mean(cross_val_score(bayes_model, X, y, cv=5, n_jobs=-1, scoring="f1"))
    
    res_bayes_model = gbrt_minimize(objective, space, n_calls=n_calls, random_state=0)
       
 
    model = KNeighborsClassifier(n_neighbors=res_bayes_model.x[0],
                                weights=res_bayes_model.x[1],
                                )
    
    model.fit(X, y)

    with open('save_models/knn_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open('save_models/knn_param.pickle', 'wb') as f:
        pickle.dump(res_bayes_model.x, f)
    
    
    return model

        
        
        
# Random Forest
def RF(X, y, n_calls=15, normalize=False):
    ''' Random Forest for Classification
    
        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_calls:    The number of GP function evaluations.
            normalize:  Scaling input data.
    '''

    
    if normalize is True: 
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X) 

    class_weights = {0: 1.0, 1: 100.0}  # Example weights (adjust as needed)        
    bayes_model = RandomForestClassifier(random_state=0, class_weight=class_weights)#, n_jobs=-1)

    
    space  = [Integer(100, 300, name='n_estimators'),
              Integer(3, 12,   name='max_depth'),
              Integer(3, 12,   name='min_samples_leaf'),
              Integer(3, 12,   name='min_samples_split'),
             ]
    
    @use_named_args(space)
    def objective(**params):
        bayes_model.set_params(**params)
        return -np.mean(cross_val_score(bayes_model, X, y, cv=5, n_jobs=-1, scoring="f1"))
    
    res_bayes_model = gbrt_minimize(objective, space, n_calls=n_calls, random_state=0)
       
 
    model = RandomForestClassifier(n_estimators=res_bayes_model.x[0],
                                   max_depth=res_bayes_model.x[1],
                                   min_samples_leaf=res_bayes_model.x[2], 
                                   min_samples_split=res_bayes_model.x[3],
                                   random_state=0,
                                   n_jobs=-1,
                                  )
    
    model.fit(X, y)

    with open('save_models/rf_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open('save_models/rf_param.pickle', 'wb') as f:
        pickle.dump(res_bayes_model.x, f)
    
    
    return model

        
        
        
# Extreme Gradient Boosting              
def XGB(X, y, n_calls=20, normalize=False):
    ''' Extreme Gradient Boosting for Classification
    
        # Arguments:
            X:          Training/Validation data matrix with inputs of size (N x Nx), Nx - number of inputs.
            y:          Training/Validation data matrix with outputs of size (N x Ny), Ny - number of outputs.
            n_calls:    The number of GP function evaluations.
            normalize:  Scaling input data.
    '''

    if normalize is True: 
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X) 
    
    class_weights = {0: 1.0, 1: 100.0}
    bayes_model = XGBClassifier(random_state=0, class_weights=class_weights, silent=True)
    
    space  = [Integer(100, 500, name='n_estimators'),
              Integer(3, 12,   name='max_depth'),
              Real(1e-3, 1,    "log-uniform", name='eta'),
              Real(0.5, 1,     "log-uniform", name='subsample'),
              Real(0.5, 1,     "log-uniform", name='colsample_bytree')
             ]
    
    @use_named_args(space)
    def objective(**params):
        bayes_model.set_params(**params)
        return -np.mean(cross_val_score(bayes_model, X, y, cv=5, n_jobs=-1, scoring="f1"))
    
    res_bayes_model = gbrt_minimize(objective, space, n_calls=n_calls, random_state=0)
       
 
    model = XGBClassifier(n_estimators=res_bayes_model.x[0],
                          max_depth=res_bayes_model.x[1],
                          eta=res_bayes_model.x[2],
                          subsample=res_bayes_model.x[3],
                          colsample_bytree=res_bayes_model.x[4], 
                          random_state=0,
                          silent=True)
    
    model.fit(X, y)

    with open('save_models/xgb_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open('save_models/xgb_param.pickle', 'wb') as f:
        pickle.dump(res_bayes_model.x, f)

    return model