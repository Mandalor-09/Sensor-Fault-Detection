import pickle
import pandas as pd
from src.exception import CustomException
import os
import joblib

def prediction(file):
    try:
        # Load preprocessor models
        df = pd.read_csv(file)
        colm = pd.read_csv(os.path.join('artifacts', 'columns', 'data.csv'), header=None)
        colm_list = colm.iloc[:, 0].tolist()

        df = df[colm_list]

        preprocessor_path_1 = os.path.join('artifacts', 'preprocessing', 'knn_imputer.joblib')
        preprocessor_path_2 = os.path.join('artifacts', 'preprocessing', 'robust_preprocessor.joblib')

        model_path = os.path.join('artifacts', 'models', 'final_model.joblib')

        model = joblib.load(model_path)
        preprocessor_1 = joblib.load(preprocessor_path_1)
        preprocessor_2 = joblib.load(preprocessor_path_2)

        # Apply transformations on the input data
        transformed_data_1 = preprocessor_1.transform(df)
        transformed_data = preprocessor_2.transform(transformed_data_1)

        # Make predictions
        predictions = model.predict(transformed_data)

        df['Output'] = predictions  # Fixed variable name

        return df
    except Exception as e:
        raise CustomException(f'Error during prediction: {e}')
