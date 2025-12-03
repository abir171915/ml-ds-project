import sys
from dataclasses import dataclass
import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os 


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self) : 
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation 
        We only use categorical features to predict average_score.
        """
        try: 
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding + scaling completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ],
                remainder="drop"   # drop all other columns (scores, etc.)
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete")
            logging.info(f"train_df shape: {train_df.shape}")
            logging.info(f"test_df shape: {test_df.shape}")

            # create average_score in BOTH dataframes
            #for df in [train_df, test_df]:
           #     df["average_score"] = (
           #         df["math score"] + df["writing score"] + df["reading score"]
           #     ) / 3

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]
            # split into input features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"X_train df shape: {input_feature_train_df.shape}")
            logging.info(f"y_train shape: {target_feature_train_df.shape}")
            logging.info(f"X_test df shape: {input_feature_test_df.shape}")
            logging.info(f"y_test shape: {target_feature_test_df.shape}")

            logging.info("Applying preprocessing object to train and test data")

            # fit on train, transform train & test
            X_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            X_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #logging.info(f"X_train_arr shape: {input_feature_train_arr.shape}")
            #logging.info(f"X_test_arr shape: {input_feature_test_arr.shape}")

            # Now concat features + target
            y_train_arr = np.array(target_feature_train_df)
            
            y_test_arr = np.array(target_feature_test_df)
            

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                X_train_arr,
                X_test_arr,
                y_train_arr,
                y_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)