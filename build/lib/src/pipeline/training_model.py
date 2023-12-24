from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataPreprocessing
from src.components.model_training import BestModel

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from dataclasses import dataclass
import os
import pickle
import joblib
import pandas as pd

from src.logger import logging
from src.exception import CustomException

def get_final_model_instance(model_name, parameters):
    if model_name == 'LogisticRegression':
        return LogisticRegression(**parameters)
    elif model_name == 'SVC':
        return SVC(**parameters)
    elif model_name == 'KNeighborsClassifier':
        return KNeighborsClassifier(**parameters)
    elif model_name == 'DecisionTreeClassifier':
        return DecisionTreeClassifier(**parameters)
    elif model_name == 'RandomForestClassifier':
        return RandomForestClassifier(**parameters)
    else:
        raise CustomException(f'Unsupported model: {model_name}')
    

def start_training():
    try:
        # Data Ingestion
        di = DataIngestion()
        raw_data_path, train_data_path, test_data_path = di.start_data_ingestion()
        logging.info(f"Data paths: {raw_data_path}, {train_data_path}, {test_data_path}")

        # Data Preprocessing
        dp = DataPreprocessing(raw_data=raw_data_path, train_data=train_data_path, test_data=test_data_path)
        processed_data = dp.pre_processing_start()
        X_train_trans, X_test_trans, y_train, y_test, knn_imputer, robust_preprocessor = processed_data
        logging.info("Data preprocessing completed successfully.")

        # Model Training and Selection
        mt = BestModel(X_train_trans, X_test_trans, y_train, y_test)
        best_model, best_params = mt.best_model_pre_start()
        logging.info(f"Best model: {best_model}")
        logging.info(f"Best parameters: {best_params}")

        # Load test data
        test_ds = test_data_path
        
        # Train the final model
        saved_model_path, knn_preprocess, robust_prepocess = train_final_model(
            best_model_name=best_model,
            best_parameters=best_params,
            X_train_trans=X_train_trans,
            X_test_trans=X_test_trans,
            y_train=y_train,
            y_test=y_test,
            test_ds=test_ds,
            knn_imputer=knn_imputer,
            robust_preprocessor=robust_preprocessor
        )

        logging.info(f"Model saved at: {saved_model_path}")

        return saved_model_path

    except CustomException as ce:
        logging.error(str(ce))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

@dataclass
class ModelConfigure():
    model_file = os.path.join('artifacts', 'models', 'final_model.joblib')

def train_final_model(best_model_name, best_parameters, X_train_trans, X_test_trans, y_train, y_test, test_ds, knn_imputer, robust_preprocessor):
    try:
        final_model = get_final_model_instance(best_model_name, best_parameters)
        final_model.fit(X_train_trans,y_train)

        y_train_prediction = final_model.predict(X_train_trans)
        train_accuracy = accuracy_score(y_train, y_train_prediction)
        logging.info(f'Train Accuracy: {train_accuracy * 100:.2f}%')

        y_test_prediction = final_model.predict(X_test_trans)
        test_accuracy = accuracy_score(y_test, y_test_prediction)
        logging.info(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        # Save the best model
        model_files = ModelConfigure()

        # Create the directory structure for the model file path
        os.makedirs(os.path.dirname(model_files.model_file), exist_ok=True)

        # Save the final model using pickle
        with open(model_files.model_file, 'wb') as mf:
            joblib.dump(final_model, mf)


        logging.info(f"The final model has been trained and saved at {ModelConfigure.model_file}")

        # Additional Testing Code
        test_ds = pd.read_csv(test_ds)
        test_df_x = test_ds.drop(columns=['Good/Bad', 'Unnamed: 0',])
        test_df_y = test_ds['Good/Bad']

        loaded_knn_imputer = joblib.load(open(knn_imputer, 'rb'))

        # Apply KNN Imputer to the new test data
        new_test_df_x = loaded_knn_imputer.transform(test_df_x)

        # Load Robust Scaler using joblib
        loaded_robust_scaler = joblib.load(open(robust_preprocessor, 'rb'))

        # Apply Robust Scaler to the new test data
        new_test_df_x = loaded_robust_scaler.transform(new_test_df_x)

        y_test_prediction = final_model.predict(new_test_df_x)
        test_accuracy = accuracy_score(test_df_y, y_test_prediction)
        logging.info(f'Test Accuracy on new test data: {test_accuracy * 100:.2f}%')

        return ModelConfigure.model_file, knn_imputer, robust_preprocessor

    except Exception as e:
        logging.error(f'Error during model training: {e}')
        raise CustomException(f'Error during model training: {e}')

if __name__ == '__main__':
    save_model_path = start_training()