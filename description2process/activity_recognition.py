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
    url = "https://www.dropbox.com/s/g05auc93ysdsdl7/best-model.pt?dl=1"  # dl=1 is important
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    with open("best-model.pt", "wb") as f :
        f.write(data)

def get_classifier():
    download_model()
    classifier = TextClassifier.load_from_file("best-model.pt")
    return classifier

classifier = get_classifier()
def contains_activity(clause, classifier = classifier) :
    sentence = Sentence(clause)
    classifier.predict(sentence)
    if str(sentence.labels[0])[0] == "0":
        return False
    else:
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
