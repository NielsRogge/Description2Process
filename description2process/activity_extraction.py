## --- Set-up environment --- ##import pandas as pd
from ast import literal_eval
import re

import spacy
spacy.cli.download("en")
nlp = spacy.load('en', disable=['tagger', 'ner'])

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('wordnet')

## --- Activity extraction with semantic role labeling --- ##
def get_argument_of_root(sent, root) :
  result = predictor.predict(sentence=sent)
  for block in result['verbs']:
    #first, search for the root
    if root == block['verb']:
      l = re.findall('\[(.*?)\]', block['description'])
      d = get_dictionary(l)
      if 'ARG1' in d :
        return True, d['ARG1']
      elif 'ARG2' in d :
        return True, d['ARG2']
      else:
        return True,""

  if result['verbs']:
    l = re.findall('\[(.*?)\]', block['description'])
    d = get_dictionary(l)

    if 'ARG1' in d:
      #set the suspected root as spacy token
      verb = d['V']
      doc3 = nlp(sent.lower())
      for token in doc3:
        if token.text == str(verb):
          suspected_root = token

      #remove determiners and possessive pronouns from the argument
      arg = d['ARG1']
      doc1 = nlp(arg)
      for t in doc1:
        if (t.dep_ == "det" or t.dep_ == "poss") and t.i == 0:
          arg = str(arg).replace(str(t) + " ", " ");
          arg = arg.lstrip()
        elif (t.dep_ == "det" or t.dep_ == "poss") and t.i !=0:
          arg = str(arg).replace(" " + str(t) + " ", " ");
          arg = arg.lstrip()

      #check for prt
      hasPrt=False
      for child in suspected_root.children:
        if child.dep_ == "prt":
          hasPrt = True
          prt = child

      if hasPrt:
        return False, suspected_root.lemma_ + " " + prt.lemma_ + " " + arg
      else:
        return False, suspected_root.lemma_ + " " + arg
    else:
      return False, ""

  else:
    return False, ""

# Function to get a dictionary from the output of SRL
def get_dictionary(l) :
    L1 = []
    L2 = []
    for item in l:
        if ":" in item:
          L1.append(re.search('(.*):',item ).group(1))
          L2.append(re.search(': (.*)',item ).group(1))
    d = dict(zip(L1,L2))
    return d

def remove_determiners(sent) :
  determiners = [' the ',' a ',' an ']
  sent = sent.lower()
  for word in determiners:
    sent = sent.replace(word, " ")
  return sent

## --- Activity extraction using active/passive rule --- ##
# Function to check passive voice
def isPassive(sent) :
    doc = nlp(sent)
    overview = []
    for token in doc:
        overview.append([token.dep_,token.lemma_])
    if ['auxpass', 'be'] in overview:
        return True
    else:
        return False

def returnActivityActive(text):
    doc = nlp(text.lower())

    hasRoot = False
    hasPrt = False
    hasDirectObject = False

    #determine the root
    for token in doc:
        if (token.dep_ == "ROOT"):
            hasRoot = True
            root = token

    #determine the prt of the root (if there is one)
    if hasRoot:
        for token in root.children:
            if token.dep_ == "prt":
                hasPrt = True
                prt = token

    #determine the direct object (if there is one)
    for child in root.children:
        if child.dep_ == "dobj":
            hasDirectObject = True
            direct_obj = child

    #determine the words to the left of the direct object (excluding determiners)
    if hasDirectObject:
        children_before_direct_obj = []
        for x in direct_obj.lefts:
            if x.dep_ != "det" and x.dep_ != "poss":
                children_before_direct_obj.append(x)

    # Check if a root was found
    if hasRoot == False:
        print("This text does not contain a root verb.")

    elif hasRoot and hasPrt == False and hasDirectObject == False:
        activity = root.lemma_

    elif hasRoot and hasPrt == False and hasDirectObject and len(children_before_direct_obj) == 0:
        #print("Root and direct object found.")
        activity = (root.lemma_ + " " + direct_obj.text)

    elif hasRoot and hasPrt == False and hasDirectObject and len(children_before_direct_obj) > 0:
        #print("Root, direct object and children of direct object found.")
        activity = (root.lemma_ + " " + ' '.join(map(str, children_before_direct_obj)) + " " + direct_obj.text)

    elif hasRoot and hasPrt and hasDirectObject and len(children_before_direct_obj) == 0:
        #print("Root, prt and direct object found.")
        activity = (root.lemma_ + " " + prt.lemma_ + " " + direct_obj.text)

    elif hasRoot and hasPrt and hasDirectObject and len(children_before_direct_obj) > 0:
        #print("Root, prt, direct object and children of direct object found.")
        activity = (root.lemma_ + " " + prt.lemma_ + " " + ' '.join(map(str, children_before_direct_obj)) + " " + direct_obj.text)

    else:
        activity ="Error: something went wrong in returnActivityActive"
    return activity

def returnActivityPassive(text):
    doc = nlp(text.lower())

    hasRoot = False
    hasPrt = False
    hasSubj = False
    hasOprd = False

    #determine the root
    for token in doc:
        if (token.dep_ == "ROOT"):
            hasRoot = True
            root = token

    #determine if present the prt of the root
    if hasRoot:
        for token in root.children:
            if token.dep_ == "prt":
                hasPrt = True
                prt = token

    #determine if present the oprd of the root
    if hasRoot:
        for token in root.rights:
            if token.dep_ == "oprd":
                hasOprd = True
                oprd = token

    #determine the subject (if there is one)
    for token in doc:
        if (token.dep_ == "nsubjpass"):
            hasSubj = True
            subj = token

    #determine the words to the left of the subject (excluding determiners and possessive pronouns)
    if hasSubj:
        children_before_subj = []
        for x in subj.lefts:
            if x.dep_ != "det" and x.dep_ != "poss":
                children_before_subj.append(x)

    #finally, return the result:
    if hasRoot == False:
        print("This text does not contain a root verb.")

    elif hasRoot and hasPrt == False and hasSubj and len(children_before_subj) == 0 and hasOprd == False:
        activity = (root.lemma_ + " " + subj.text)

    elif hasRoot and hasPrt == False and hasSubj and len(children_before_subj) == 0 and hasOprd:
        activity = (root.lemma_ + " " + subj.text + " " + oprd.text)

    elif hasRoot and hasPrt == False and hasSubj and len(children_before_subj) > 0:
        activity = (root.lemma_ + " " + ' '.join(map(str, children_before_subj)) + " " + subj.text)

    elif hasRoot and hasPrt and hasSubj and len(children_before_subj) == 0:
        activity = (root.lemma_ + " " + prt.lemma_ + " " + subj.text)

    elif hasRoot and hasPrt and hasSubj and len(children_before_subj) > 0:
        activity = (root.lemma_ + " " + prt.lemma_ + " " + ' '.join(map(str, children_before_subj)) + " " + subj.text)

    else:
        activity = "Error: something went wrong in returnActivityPassive()"
    return activity

def get_activity_active_passive(sent):
    if isPassive(sent):
        return returnActivityPassive(sent)
    else:
        return returnActivityActive(sent)

def get_activity_SRL(sent):
  doc = nlp(sent.lower())
  for token in doc:
    if token.dep_ == "ROOT":
      is_arg, arg = get_argument_of_root(sent.lower(), str(token))
      #first, remove any determiners (such as "the", "a" etc.) and possessive pronouns from the argument :
      doc1 = nlp(arg)
      for t in doc1:
        if t.dep_ == "det" or t.dep_ == "poss":
          arg = str(arg).replace(str(t) + " ", "");
          #print("are we here")
          arg = arg.lstrip()

      #second, determine the activity:

      #determine if present the prt of the root (for verbs like back UP, fill IN, etc.)
      hasPrt=False
      for child in token.children:
        if child.dep_ == "prt":
          hasPrt = True
          prt = child

      if is_arg and hasPrt == False:
        activity = (str(token.lemma_) + " " + arg)
      elif is_arg:
        activity = (str(token.lemma_) + " " + prt.lemma_ + " " + arg)
      else:
        activity = arg

      return activity

## --- Activity extraction combined --- ##
def get_activity(sent):
    activity_SRL = get_activity_SRL(sent)
    return get_activity_active_passive(sent) if not activity_SRL else activity_SRL

def get_activity_df(labeled_clauses):
    labeled_clauses['activity'] = labeled_clauses.loc[labeled_clauses["label"] == True].apply(lambda row: get_activity(row['clause']), axis = 1)
    return labeled_clauses
