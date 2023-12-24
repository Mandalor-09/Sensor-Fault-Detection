Run commend "python setup.py install" to built initial setup of the application.
Run commend "python src/pipeline/training_model.py run" to built initial setup of the application.
Run "python app.py" to run the flask server.

Brief: In electronics, a wafer (also called a slice or substrate) is a thin slice of semiconductor, such as a crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. The wafer serves as the substrate(serves as foundation for contruction of other components) for microelectronic devices built in and upon the wafer.

It undergoes many microfabrication processes, such as doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as an integrated circuit.


The Project is mainly divided into 2 parts 
1)Components
2)Pipeline

Components has 3 file 
1)Data Ingestion
2)Data Processing
3)Model Training 

Pipeline has 2 files 
1)Final Model Building
2)Prediction

1)Data Ingestion: This part will be responsible for getting data from csv file sources and spliting the data into train data,test data,validation data.

2)Data Preprocessing -clean_data(df): This function takes a dataframe and remove all the missing values, outliers and duplicates from it and then standardizes the data and make it avaiable for training.

3)Model Training: 
This module will take data from Data Ingestion and process it using Data Processing module. It will try to find the Best Model and Best parameter for the model using Hyper tunning.

4)Final Model Building:
In this step, we will build a machine learning model using scikit-learn library and store the model using joblib at "artifacts/models/final_model.joblib".

5)Prediction
This file contain function related to Prediction of Output.
