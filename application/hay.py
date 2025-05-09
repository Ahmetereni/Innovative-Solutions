# hay.py
from flask import jsonify
from flask import Blueprint, render_template, session, request
import pandas as pd
from flask import render_template
import pandas as pd
import json


hay = Blueprint('hay', __name__)


@hay.route("/view-patients")
def search():

    df = pd.read_csv('10-patients.csv')

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

    return render_template('search.html',data=data)

