import os
import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'credit_card_fraud_model.pkl'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to store model and scaler
model = None
scaler = None

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, tuple) and len(data) == 2:
                    model, scaler = data
                else:
                    model = data
                    scaler = None
            print("[INFO] Model and Scaler loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            model, scaler = None, None

# Load model on startup
load_model()

@app.route('/')
def index():
    return render_template('index.html', model_exists=(model is not None))

@app.route('/train', methods=['POST'])
def train():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"})

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        total, fraud, accuracy = train_internal(file_path)
        load_model()

        return jsonify({
            "success": True, 
            "message": "Model trained successfully!", 
            "total": int(total), 
            "fraud": int(fraud),
            "accuracy": f"{accuracy:.2f}%"
        })
    except Exception as e:
        print(f"[TRAIN ERROR] {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None:
        return jsonify({"success": False, "message": "Model not trained yet. Please train the model first."})

    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "message": "No selected file"})

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"predict_{file.filename}")
        file.save(file_path)

        df = pd.read_csv(file_path)
        
        required_features = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                             "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                             "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
        
        if not all(col in df.columns for col in required_features):
             return jsonify({"success": False, "message": "CSV missing required features (Time, V1-V28, Amount)"})

        X = df[required_features]
        X_scaled = scaler.transform(X) if scaler else X
        predictions = model.predict(X_scaled)
        
        fraud_count = np.sum(predictions == 1)
        legit_count = np.sum(predictions == 0)
        total_count = len(predictions)

        return jsonify({
            "success": True,
            "type": "prediction",
            "total": int(total_count),
            "fraud": int(fraud_count),
            "legit": int(legit_count)
        })
    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        return jsonify({"success": False, "message": str(e)})

def train_internal(file_path):
    df = pd.read_csv(file_path)
    if "Class" not in df.columns:
        raise ValueError("Training data must contain 'Class' column.")

    required_features = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                         "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                         "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    
    df = df.dropna()
    
    # OPTIMIZATION: Subsample if data is large (to speed up web-based training)
    # If dataset > 20,000 rows, we take a random sample of 20,000
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)

    X = df[required_features]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    local_scaler = StandardScaler()
    X_train_scaled = local_scaler.fit_transform(X_train)
    X_test_scaled = local_scaler.transform(X_test)

    # n_jobs=-1 uses all CPU cores for faster training
    local_model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    local_model.fit(X_train_scaled, y_train)

    accuracy = local_model.score(X_test_scaled, y_test) * 100
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((local_model, local_scaler), f)

    return len(df), df['Class'].sum(), accuracy

if __name__ == '__main__':
    app.run(debug=True, port=5001)
