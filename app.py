from flask import Flask, render_template, request, send_file, jsonify
from src.pipeline.prediction import prediction
from src.logger import logging
from src.exception import CustomException
import os
import sys

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@app.route('/')
def hello():
    return render_template('index.html', name='World', download_link=None)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)


@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                return render_template('index.html', error='No file part')

            file = request.files['file']

            # If the user does not select a file, the browser also submits an empty part without filename
            if file.filename == '':
                return render_template('index.html', error='No selected file')

            # Perform prediction
            prediction_result = prediction(file)

            if not prediction_result.empty:
                logging.info("Prediction completed. Preparing DataFrame for download.")

                # Save DataFrame to a temporary CSV file
                temp_csv_file = 'temp_prediction_result.csv'
                prediction_result.to_csv(temp_csv_file, index=False)

                logging.info(f"Saved DataFrame to {temp_csv_file}. Downloading prediction file.")

                # Provide the download link in the template
                download_link = f'/download/{temp_csv_file}'
                
                return render_template('index.html', download_link=download_link)
            else:
                # Log the error and return a JSON response
                logging.error("Prediction failed: DataFrame is empty.")
                return jsonify({'error': 'Prediction failed: DataFrame is empty'}), 500

        else:
            return render_template('index.html')

    except Exception as e:
        # Log unexpected errors and raise a custom exception
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(debug=True)
