from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__,static_url_path="/static")

# Load the TensorFlow SavedModel
model_path = 'Best_model.hdf5'  # Replace this with the path to your TensorFlow SavedModel
model = tf.keras.models.load_model(model_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    csv_file = request.files['file']
    if not csv_file:
        return jsonify({"error": "No CSV file provided."}), 400

    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file)
    colms = [colm for colm in df.columns if colm not in ['date', 'id', "sales", "year"]]
    X_train = df[colms]
    Y_train = df['sales']
    X_train_array = X_train.values
    Y_train_array = Y_train.values
    
    time_steps = 1
    num_features = X_train_array.shape[1]
    Xtr = X_train_array.reshape((X_train_array.shape[0], time_steps, num_features))

    
    train_predictions = model.predict(Xtr).flatten()

    
    def numpy_to_list(numpy_array):
        if isinstance(numpy_array, np.ndarray):
            return numpy_array.tolist()
        elif isinstance(numpy_array, np.floating):
            return float(numpy_array)
        elif isinstance(numpy_array, (np.int64, np.int32, int)):
            return int(numpy_array)
        return numpy_array

    predictions_to_display = [numpy_to_list(val) for val in np.expm1(train_predictions)[:5]]

    # Save predictions as CSV file (encode as UTF-8)
    train_results = pd.DataFrame(data={'Train Predictions': np.expm1(train_predictions), 'Actuals': np.expm1(Y_train_array)})
    predictions_csv = train_results.to_csv(index=False, encoding='utf-8')

    # Return the predictions as JSON response
    return jsonify({"predictions_to_display": predictions_to_display, "predictions_csv": predictions_csv}), 200


@app.route('/download_predictions_csv')
def download_predictions_csv():
    predictions_csv = request.args.get('predictions_csv')
    if predictions_csv:
        return send_file(io.BytesIO(predictions_csv.encode()), mimetype='text/csv', attachment_filename='predictions.csv', as_attachment=True)
    return jsonify({"error": "CSV data not found."}), 404


if __name__ == '__main__':
    app.run(debug=True)
