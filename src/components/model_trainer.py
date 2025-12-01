import os 
import sys
from dataclasses import dataclass

from catboost import catBoostRegressor
from sklearn.ensemble import (
    adaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboosti import XGBRegressor

from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object

@dataclass 
class ModelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_tainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            ) 
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Liner Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Classifier" : CatBoostRegressor(verbose = False),
                "AdaBoost Classifier" : AdaBoostRegressor(),
            }

            model_report : dict = evaluate_model( X_train = X_train, y_trian = y_trian, X_test= X_test, y_test = y_test, models = models)
        except: