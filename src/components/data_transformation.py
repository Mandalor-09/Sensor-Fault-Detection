from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
import pickle
import joblib
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion

print("pandas version:", pd.__version__)

@dataclass
class DataPreprocessorConfigure:
    knn_imputer = os.path.join('artifacts', 'preprocessing', 'knn_imputer.joblib')
    robust_preprocessor = os.path.join('artifacts', 'preprocessing', 'robust_preprocessor.joblib')

def col_with_variance_0(df):
    return [col for col in df.columns if df[col].dtype != 'O' and df[col].std() == 0]

def get_redundant_cols(df: pd.DataFrame, missing_thresh=0.7):
    cols_missing_ratios = df.isna().sum().div(df.shape[0])
    return list(cols_missing_ratios[cols_missing_ratios > missing_thresh].index)

def dropping_columns_on_basis_of_correlation(df):
    columns_to_drop = set()
    threshold = 0.9
    relation = df.corr()
    for columns in range(len(relation.columns)):
        for rows in range(columns):
            if abs(relation.iloc[columns, rows]) > threshold:
                col_name = relation.columns[columns]
                columns_to_drop.add(col_name)
    return list(columns_to_drop)

def feature_scaling_df(df):
    try:
        # Exclude non-numeric columns
        cols_to_drop_1 = get_redundant_cols(df)
        cols_to_drop_2 = col_with_variance_0(df)

        numeric_cols = df.select_dtypes(include=['number']).columns

        df_numeric = df[numeric_cols]

        cols_to_drop_3 = dropping_columns_on_basis_of_correlation(df_numeric)

        return list(set(cols_to_drop_1 + cols_to_drop_2 + cols_to_drop_3))

    except Exception as e:
        logging.error(f'Error during feature scaling: {e}')
        raise CustomException(f'Error during feature scaling: {e}')



class DataPreprocessing:
    def __init__(self, raw_data, train_data, test_data):
        self.raw_data = raw_data
        self.train_data = train_data
        self.test_data = test_data
        self.preprocess_config = DataPreprocessorConfigure()

    def pre_processing_start(self):
        try:
            logging.info('Data Transformation Step initiated')
            #logging.info(os.getcwd(),self.raw_data,'<<<<<<<<<<<<<<<<<<<<>>>>>>>>>')

            df = pd.read_csv(self.raw_data)
            logging.info('Read raw data successfully')

            print(df.head(2))
            
            columns_to_drop = feature_scaling_df(df)
            final_df = df.drop(columns=columns_to_drop)

            logging.info('Train and Test Data Splitting Started')
            train = final_df.iloc[:90, :].drop(columns=['Unnamed: 0','Unnamed: 0.1'])
            print(self.train_data,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            train.to_csv(self.train_data)
            logging.info(f'Saved training data to {self.train_data}')
            test = final_df.iloc[90:, :].drop(columns=['Unnamed: 0','Unnamed: 0.1'])
            test.to_csv(self.test_data)
            logging.info(f'Saved testing data to {self.test_data}')

            logging.info('Preprocessing Step Initiated')
            
            x_columns = train.drop(columns=['Good/Bad']).columns

            # Create the directory if it doesn't exist
            os.makedirs(os.path.join('artifacts', 'columns'), exist_ok=True)

            # Save x_columns to a CSV file
            x_columns.to_series().to_csv(os.path.join('artifacts', 'columns', 'data.csv'), header=False, index=False)
            
            logging.info(f'Columns after dropping unnecessary columns: {x_columns}')
            X = train.drop(columns=['Good/Bad'])
            y = train['Good/Bad']

            pipeline_preprocessing_1 = Pipeline(
                steps=[
                    ('Imputer', KNNImputer(n_neighbors=3))
                ]
            )

            new_train = pipeline_preprocessing_1.fit_transform(X)
            logging.info('KNN imputation completed successfully')

            # Save KNN Imputer using joblib
            os.makedirs(os.path.dirname(self.preprocess_config.knn_imputer), exist_ok=True)
            with open(self.preprocess_config.knn_imputer, 'wb') as f:
                joblib.dump(pipeline_preprocessing_1, f)

            logging.info(f'Saved KNN imputer to {self.preprocess_config.knn_imputer}')

            # SMOTE
            smote = SMOTE(sampling_strategy='all', k_neighbors=2)
            train_X, train_y = smote.fit_resample(new_train, y)

            X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

            pipeline_preprocessing_2 = Pipeline(
                steps=[
                    ('Scaling', RobustScaler())
                ]
            )

            X_train_trans = pipeline_preprocessing_2.fit_transform(X_train)
            X_test_trans = pipeline_preprocessing_2.transform(X_test)

            # Save Robust Scaler using joblib
            os.makedirs(os.path.dirname(self.preprocess_config.robust_preprocessor), exist_ok=True)
            with open(self.preprocess_config.robust_preprocessor, 'wb') as f:
                joblib.dump(pipeline_preprocessing_2, f)

            logging.info(f'Saved Robust Scaler to {self.preprocess_config.robust_preprocessor}')
            logging.info('Data Ingestion and Transformation Step Completed')

            return [
                X_train_trans, X_test_trans, y_train, y_test,
                self.preprocess_config.knn_imputer, self.preprocess_config.robust_preprocessor
            ]

        except Exception as e:
            logging.error(f'Error during preprocessing: {e}')
            raise CustomException(f'Error during preprocessing: {e}')

if __name__ == '__main__':
    di = DataIngestion()
    data_paths = di.start_data_ingestion()
    print(data_paths, '<<<<<<<<<<<<<<<>>>>>>>>>>>')

    dp = DataPreprocessing(raw_data=data_paths[0], train_data=data_paths[1], test_data=data_paths[2])
    processed_data = dp.pre_processing_start()
    print(processed_data, '<<<<<<<<<<<<<<<>>>>>>>>>>>')
