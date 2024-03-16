from flask import Flask, request, jsonify
from model import My_Classifier_Model

app = Flask(__name__)
model = My_Classifier_Model()

@app.route('/train', methods=['POST'])
def train_model():
    dataset_filename = request.json['dataset']
    model.train(dataset_filename)
    return jsonify({'message': 'Model training completed.'})

@app.route('/predict', methods=['POST'])
def predict_model():
    dataset_filename = request.json['dataset']
    model_name = request.json['model']
    model.predict(dataset_filename, model_name)
    return jsonify({'message': 'Prediction completed. Result saved in ./data/result.csv'})

@app.route('/tune', methods=['POST'])
def tune_hyperparameters():
    dataset_filename = request.json['dataset']
    model.tune_hyperparameters(dataset_filename)
    return jsonify({'message': 'Hyperparameter tuning completed.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
