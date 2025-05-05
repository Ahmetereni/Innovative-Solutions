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


def tutorial1_basic_qa_pipeline(question,username):
    import logging

    logging.basicConfig(
        format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
    logging.getLogger("haystack").setLevel(logging.INFO)

    from haystack.document_stores import ElasticsearchDocumentStore
    from haystack.nodes import FARMReader, BM25Retriever

    # launch_es()

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(
        host="localhost", username="elastic", password="WbLoke8xGtKNRu*RPdjd", index=f"{username}")



    retriever = BM25Retriever(document_store=document_store)

    reader = FARMReader(
        model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    from haystack.pipelines import ExtractiveQAPipeline

    pipe = ExtractiveQAPipeline(reader, retriever)

    # Voilà! Ask a question!
    prediction = pipe.run(
        query=f"{question}", params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 3}}
    )

    # print(prediction["answers"][0].meta["name"])

    return prediction["answers"]


# if __name__ == "__main__":
#     tutorial1_basic_qa_pipeline()
