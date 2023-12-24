from dataclasses import dataclass
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

print("pandas version:", pd.__version__)
print(os.getcwd() + r'\notebook\wafer.csv')
@dataclass
class Data_ingestion_initialize_file():
    raw_ds_path = os.path.join('artifacts','ds','raw_ds.csv')
    train_ds_path = os.path.join('artifacts','ds','train_ds.csv')
    test_ds_path = os.path.join('artifacts','ds','test_ds.csv')

class DataIngestion():
    def __init__(self):
        self.initializing_ds = Data_ingestion_initialize_file()

    def start_data_ingestion(self):
        logging.info('Data ingestion Method Initiated')
        
        file = os.getcwd() + r'/notebook/wafer.csv'
        
        try:
            df = pd.read_csv(file)

            if not df.empty:
                print(df.head(2))
                logging.info('Read Data Success')

                os.makedirs(os.path.dirname(self.initializing_ds.raw_ds_path), exist_ok=True)
                df.to_csv(self.initializing_ds.raw_ds_path)
                logging.info("Raw dataset saved successfully")

                os.makedirs(os.path.dirname(self.initializing_ds.train_ds_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.initializing_ds.test_ds_path), exist_ok=True)
                logging.info('All data directory made successfully')

                return [self.initializing_ds.raw_ds_path, self.initializing_ds.train_ds_path, self.initializing_ds.test_ds_path]
            else:
                logging.info('Read Data Un Success')
        except FileNotFoundError as e:
            logging.error(f'File not found: {file}')
            logging.error(e, exc_info=True)
        except Exception as e:
            logging.error(f'Error Occurred in Data Ingestion: {str(e)}')

if __name__ == '__main__':
    di = DataIngestion()
    a = di.start_data_ingestion()
    print(a, '<<<<<<<<<<<<<<<>>>>>>>>>>>')
