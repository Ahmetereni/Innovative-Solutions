import os
from flask import Blueprint, render_template, session, send_file, request, redirect, url_for, flash
from flask_login import login_required
from path import mypath
from werkzeug.utils import secure_filename
from application.dochandler import preprocessor

main = Blueprint('main', __name__)

# setting up directory
ABSOLUTE_PATH = mypath()
ALLOWED_EXTENSIONS = {'pdf', 'docx', "txt"}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import json
import random


with open('test_cases.json','r') as f:
    test_cases=json.load(f)

# ======================= DATASET =======================
# load
data = pd.read_csv('full_transplant_monitoring_dataset.csv')

# define features
drop_features = [
    'BiopsyResult', 'RejectionTime(Hrs)', 'Infection', 'Complications',
    'SuccessfulTransplant', 'Intervention', 'Time', 'TransplantNeeded', 'TransplantYear'
]

target_column = 'RejectionStatus'

# split x and y
X = data.drop(columns=[target_column] + drop_features)
y = data[target_column]

# define categorical and numeric features
categorical_features = ['Gender', 'TransplantType']
numeric_features = [col for col in X.columns if col not in categorical_features]

# build preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# build the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ======================= TRAINING =======================
# train-test split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = pipeline.fit(X_train, y_train)

# evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# print(f"Model accuracy on test set: {acc:.2%}")

# specify expected features for prediction
expected_features = X.columns.tolist()
# print("Expected features:", expected_features)
from flask_cors import CORS
from flask import render_template
# ======================= FLASK =======================

@main.route("/")
def indexpage():

    df = pd.read_csv('full_transplant_monitoring_dataset.csv')

    # Columns to keep
    columns_to_keep = [
    'Age', 'Gender', 'BloodPressure', 'HeartAttack', 'Diabetes',
    'TransplantType', 'CRP', 'IL-6', 'WBC', 'ESR', 'Creatinine',
    'Hemoglobin', 'Cholesterol', 'BloodSugar', 'PlateletCount', 'ALT', 'eGFR'
    ]

    # Select only the needed columns
    df_filtered = df[columns_to_keep]
    wrapped_data = [{"input": row} for row in df_filtered.to_dict(orient="records")]
    data=(wrapped_data)
    # Save to a new CSV

    return render_template('index.html',data=data)


@main.route("/real-time-patients")
def real_time_patients():
    return render_template('real_time.html')

@main.route('/test')
def test():
    # # Expect JSON like: { "input": { "feature1": value1, ... } }
    global test_cases
    confidence_list=[]

    for i in range(10):      
        input_data=test_cases[i]
        input_data["CRP"] = round(random.uniform(0.1, 3.0), 1)
        input_data["IL-6"] = round(random.uniform(5.0, 10.0), 1)
        input_data["WBC"] = round(random.uniform(4.0, 10.0), 1)
        input_data["BloodPressure"] = round(random.uniform(100, 110), 1)
        input_data["BloodSugar"] = round(random.uniform(70, 110), 1)
        input_data["Hemoglobin"] = round(random.uniform(13.0, 16.0), 1)
           
        if input_data is None:
            return jsonify({'error': 'Invalid input format. Expected JSON with key "input".'}), 400

        # Validate input
        # Vvalidate input data
        missing = [feat for feat in expected_features if feat not in input_data]
        if missing:
            return jsonify({'error': f'Missing required features: {missing}'}), 400

        # drop unwanted features
        for feat in drop_features + [target_column]:
            input_data.pop(feat, None)

        # convert to DataFrame
        input_df = pd.DataFrame([input_data])
        try:
            # get probabilities and prediction
            probabilities = model.predict_proba(input_df)[0]
            prob_doing_well = float(probabilities[0])

            # predict and label
            pred = model.predict(input_df)[0]
            prediction_label = "At Risk" if pred == 1 else "Doing Well"
            confidence = {
                "Rating": float("{:.2f}".format(prob_doing_well)),
                "Prediction":prediction_label
            }

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        confidence_list.append(confidence)

    # response
    return jsonify({
        'confidence_list': confidence_list,
    })




@main.route('/submit-patient', methods=['POST'])
def predict():

    # # Expect JSON like: { "input": { "feature1": value1, ... } }
    req_data = request.get_json(force=True)
  
#     input_data = req_data.get('input')[0]
#     with open('test_cases.json','r') as f:
#         input_data=json.load(f)

    input_data= {
    "input": {
      "Age": 57,
      "Gender": "Male",
      "BloodPressure": 133.9,
      "HeartAttack": 1,
      "Diabetes": 1,
      "TransplantType": "Kidney",
      "CRP": 7.3,
      "IL-6": 13.1,
      "WBC": 9.4,
      "ESR": 38.5,
      "Creatinine": 2.1,
      "Hemoglobin": 11.8,
      "Cholesterol": 208.4,
      "BloodSugar": 157.5,
      "PlateletCount": 337.2,
      "ALT": 53.9,
      "eGFR": 33.6
    }
  }
    print("request:",req_data["input"])
    print()
#     # print("innnnnn",input_data)
    input_data=req_data['input']
    print("beklenen: ",input_data)
    # input_data=req_data

    if input_data is None:
        return jsonify({'error': 'Invalid input format. Expected JSON with key "input".'}), 400

    # Validate input
     # Vvalidate input data
    missing = [feat for feat in expected_features if feat not in input_data]
    if missing:
        print("-missssssssss")
        return jsonify({'error': f'Missing required features: {missing}'}), 400

    # drop unwanted features
    for feat in drop_features + [target_column]:
        input_data.pop(feat, None)

    # convert to DataFrame
    input_df = pd.DataFrame([input_data])
    try:
        # get probabilities and prediction
        probabilities = model.predict_proba(input_df)[0]
        prob_doing_well = float(probabilities[0])
        prob_at_risk = float(probabilities[1])
        confidence = {
            "At Risk": round(prob_at_risk, 2),
            "Doing Well": round(prob_doing_well, 2)
        }

        # predict and label
        pred = model.predict(input_df)[0]
        prediction_label = "At Risk" if pred == 1 else "Doing Well"
    except Exception as e:
        print(str(e))
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # print("Confidence",confidence)
    # print("Confidence",prediction_label)

    # response
    return jsonify({
        'confidence': confidence,
        'prediction': prediction_label
    })
if __name__ == '__main__':
    main.run(debug=True)


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/view/<path:filename>')
@login_required
def send_attachment(filename):
    strusername = str(session["username"])
    userfile = os.path.join(ABSOLUTE_PATH, 'static', 'users', strusername, filename)
    return send_file(userfile)


@main.route('/view', methods=['GET', 'POST'])
def view():
    strusername = str(session["username"])
    users_dir = os.path.join(ABSOLUTE_PATH, 'static', 'users', strusername)

    # Ensure the user's directory exists
    if not os.path.exists(users_dir):
        os.makedirs(users_dir)

    upload_files = request.files.getlist('file')

    if request.method == 'POST':
        for file in upload_files:
            filename = file.filename
            if allowed_file(filename):
                file_path = os.path.join(users_dir, secure_filename(file.filename))
                file.save(file_path)
                
                # Process the uploaded file
                extension = filename.rsplit('.', 1)[1].lower()
                processed_filepath = os.path.join(users_dir, filename)
                # preprocessor(filepath=processed_filepath,
                #              filename=filename, username=strusername, extension=extension)

        return render_template('view.html', users_dir=os.listdir(users_dir), dirname=strusername)

    return render_template('view.html', users_dir=os.listdir(users_dir), dirname=strusername)


@main.route('/delete/<path:filename>', methods=['GET'])
@login_required
def delete(filename):
    # from haystack.document_stores import ElasticsearchDocumentStore
    strusername = str(session["username"])
    # document_store = ElasticsearchDocumentStore(
    #     host="localhost", username="elastic", password="WbLoke8xGtKNRu*RPdjd", index=f"{strusername}")

    userfile = os.path.join(ABSOLUTE_PATH, 'static', 'users', strusername, filename)

    filters = {"name": filename}
    try:
        # Remove file from filesystem
        if os.path.exists(userfile):
            os.remove(userfile)
        else:
            flash("File not found", "error")
            return redirect(url_for('main.view'))

        # Remove document from Elasticsearch
        # document_store.delete_documents(index=strusername, filters=filters)
        
        flash("File deleted successfully", "success")
        return redirect(url_for('main.view'))

    except FileNotFoundError:
        flash("Error: File not found", "error")
    except PermissionError:
        flash("Error: Permission denied", "error")
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
    
    return redirect(url_for('main.view'))
