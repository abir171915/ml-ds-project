import os
import sys
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try: 
        report = {}

        for name, mdl in models.items():
            #model training
            mdl.fit(X_train, y_train)

            #prediction
            y_train_pred = mdl.predict(X_train)
            y_test_pred = mdl.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

        return report

    except Exception as e: 
        raise CustomException(e, sys)