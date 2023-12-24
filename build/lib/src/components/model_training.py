# Import necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataPreprocessing

@dataclass
class ModelConfigure:
    pass

class BestModel:
    def __init__(self, X_train_trans, X_test_trans, y_train, y_test):
        self.X_train_trans, self.X_test_trans, self.y_train, self.y_test = X_train_trans, X_test_trans, y_train, y_test

    def best_model_pre_start(self):
        # Define parameter grids for each model
        param_grid_lr = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }

        param_grid_svc = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }

        param_grid_knn = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }

        param_grid_dt = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Create a list of tuples where each tuple contains a model and its corresponding parameter grid
        models = [
            (LogisticRegression(), param_grid_lr),
            (SVC(), param_grid_svc),
            (KNeighborsClassifier(), param_grid_knn),
            (DecisionTreeClassifier(), param_grid_dt),
            (RandomForestClassifier(), param_grid_rf)
        ]

        # Perform GridSearchCV for each model
        details = {}
        try:
            for model, param_grid in models:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
                grid_search.fit(self.X_train_trans, self.y_train)

                logging.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
                logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
                details[model.__class__.__name__] = {
                    'Best parameters': grid_search.best_params_,
                    'Best cross-validation score': round(grid_search.best_score_, 4)
                }

            max_score = -1
            best_model = None
            best_parameters = None

            for model_name, model_details in details.items():
                cross_val_score = model_details['Best cross-validation score']

                if cross_val_score > max_score:
                    max_score = cross_val_score
                    best_model = model_name
                    best_parameters = model_details['Best parameters']

            logging.info(f"The maximum cross-validation score is {max_score:.4f} for the model {best_model}.")
            logging.info(f"The best parameters for {best_model} are: {best_parameters}")

            return best_model, best_parameters

        except Exception as e:
            logging.error(f"Error in model selection: {str(e)}")
            raise CustomException("Model selection failed.")

if __name__ == '__main__':
    try:
        # Data Ingestion
        di = DataIngestion()
        data_paths = di.start_data_ingestion()
        logging.info(f"Data paths: {data_paths}")

        # Data Preprocessing
        dp = DataPreprocessing(raw_data=data_paths[0], train_data=data_paths[1], test_data=data_paths[2])
        processed_data = dp.pre_processing_start()
        logging.info("Data preprocessing completed successfully.")

        X_train_trans, X_test_trans, y_train, y_test, knn_imputer, robust_preprocessor = processed_data[0] ,processed_data[1],processed_data[2],processed_data[3],processed_data[4],processed_data[5]


        # Model Training and Selection
        mt = BestModel(X_train_trans, X_test_trans, y_train, y_test)
        best_model, best_params = mt.best_model_pre_start()
        logging.info(f"Best model: {best_model}")
        logging.info(f"Best parameters: {best_params}")

    except CustomException as ce:
        logging.error(str(ce))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
