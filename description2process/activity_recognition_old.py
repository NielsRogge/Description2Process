from tensorflow.contrib import predictor
import os, sys
import urllib.request
import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence

def determine_path ():
    try:
        root = __file__
        if os.path.islink (root):
            root = os.path.realpath (root)
        return os.path.dirname (os.path.abspath (root))
    except:
        print ("I'm sorry, but something is wrong.")
        print ("There is no __file__ variable. Please contact the author.")
        sys.exit ()

def check_models_in_package ():
    print ("module is running")
    print (determine_path ())
    print ("My various models are:")
    files = [f for f in os.listdir(determine_path () + "/models")]
    print (files)

def download_model():
    url = "https://www.dropbox.com/s/6zqqq7a1ef9ln96/USE%202.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

def get_model():
    saved_model_predictor = predictor.from_saved_model("./USE")
    return saved_model_predictor

classifier = get_model()
def contains_activity(clause) :
    content_tf_list = tf.train.BytesList(value=[clause.encode('utf-8')])
    clause = tf.train.Feature(bytes_list=content_tf_list)
    clause_dict = {'clause': clause}
    features = tf.train.Features(feature=clause_dict)
    example = tf.train.Example(features=features)
    serialized_example = example.SerializeToString()
    output_dict = saved_model_predictor({'inputs': [serialized_example]})

    if output_dict['scores'][0][1] > 0.5 :
        return True
    else:
        return False

def contains_activity_list(clauses):
    labels = []
    for clause in clauses:
        labels.append(contains_activity(clause))
    return pd.DataFrame({"clause": clauses,"label":labels})

def contains_activity_df(df):
    description_id = []
    clauses = []
    id = 0
    for index, row in df.iterrows():
        clauses.extend( row.loc['clauses'] )
        description_id.extend( [id] * len(row.loc['clauses']) )
        id = id+1

    labeled_clauses = pd.DataFrame({"id_description": description_id, "clause": clauses})
    labeled_clauses['label'] = labeled_clauses.apply(lambda row: contains_activity(row['clause']), axis = 1)

    return labeled_clauses
