import urllib.request
from tensorflow.contrib import predictor
import tensorflow as tf
import os, sys
import requests, zipfile, io

def download_model():
    url = "https://www.dropbox.com/s/iy3oxx4eq7107s7/USE_bis.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

def get_model():
    classifier = predictor.from_saved_model("USE_bis")
    return classifier

download_model()
classifier = get_model()

def contains_activity(clause):
  content_tf_list = tf.train.BytesList(value=[clause.encode('utf-8')])
  clause = tf.train.Feature(bytes_list=content_tf_list)
  clause_dict = {'clause': clause}
  features = tf.train.Features(feature=clause_dict)
  example = tf.train.Example(features=features)
  serialized_example = example.SerializeToString()
  output_dict = classifier({'inputs': [serialized_example]})
  if output_dict['scores'][0][0] > 0.5 :
    return False
  else :
    return True

def contains_activity_list(clauses, classifier = classifier):
    labels = []
    for clause in clauses:
        labels.append(contains_activity(clause, classifier))
    return pd.DataFrame({"clause": clauses,"label":labels})

def contains_activity_df(df, classifier = classifier):
    description_id = []
    clauses = []
    id = 0
    for index, row in df.iterrows():
        clauses.extend( row.loc['clauses'] )
        description_id.extend( [id] * len(row.loc['clauses']) )
        id = id+1

    labeled_clauses = pd.DataFrame({"id_description": description_id, "clause": clauses})
    labeled_clauses['label'] = labeled_clauses.apply(lambda row: contains_activity(row['clause'], classifier), axis = 1)

    return labeled_clauses
