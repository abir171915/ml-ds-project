import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.exception import CustomException 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    '''
    this funtion is responsible for choosing best model
    and this funtion returns a dictionary where model name and their trian score is saved
    
    '''
    try: 
        report = {}

        for name, mdl in models.items():
            print(f"\nTuning hyperparameters for {name}")
            param_grid = params.get(name, {})

            if param_grid:
                gs = GridSearchCV(
                    estimator = mdl,
                    param_grid = param_grid,
                    cv = 3,
                    scoring = "r2",
                    n_jobs = -1,
                    verbose = 0,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = mdl
                best_model.fit(X_train, y_train)
            #model training
            #mdl.fit(X_train, y_train)

            #prediction
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            print(f"{name} - Train R2 {train_model_score: .4f}, Test R2 {test_model_score: .4f}")

        return report

    except Exception as e: 
        raise CustomException(e, sys)