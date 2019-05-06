# used for constituent parsing
import benepar
benepar.download('benepar_en')

# used to create a tree structure
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tree import ParentedTree

# document parser
import spacy
import spacy.cli
spacy.cli.download("en")
from benepar.spacy_plugin import BeneparComponent
from nltk import Tree

import string
import pandas as pd
import sys

def get_clauses_df(descriptions):
    nlp = spacy.load('en')
    nlp.add_pipe(BeneparComponent("benepar_en"))

    descriptions = descriptions.assign(clauses = "")
    for i in range(len(descriptions)) :
        text = descriptions.iloc[i,0]
        text = text.replace(";",".")
        text = text.replace("\n"," ")
        doc = nlp(text)

        subsentences = []
        for sent in doc.sents :
            subtexts = create_tree(sent)
            subtexts = lower_no_punct(subtexts)
            subsent1 = remove_double_subsents(subtexts)
            subsent1 = reorder_subsents(sent.text, subsent1)
            subsent2 = concatenate_sep_words(subsent1)
            final = capitalize_first_letters(subsent2)
            subsentences = subsentences + final

        descriptions.at[i,'clauses'] = subsentences

    return descriptions

def get_clauses(text):
    nlp = spacy.load('en')
    nlp.add_pipe(BeneparComponent("benepar_en"))

    text = text.replace(";",".")
    text = text.replace("\n"," ")
    doc = nlp(text)

    subsentences = []
    for sent in doc.sents :
        subtexts = create_tree(sent)
        subtexts = lower_no_punct(subtexts)
        subsent1 = remove_double_subsents(subtexts)
        subsent1 = reorder_subsents(sent.text, subsent1)
        subsent2 = concatenate_sep_words(subsent1)
        final = capitalize_first_letters(subsent2)
        subsentences = subsentences + final

    return subsentences

# input a spacy document and the number of the sentence to be parsed.
def check_const_parsing(doc, num_sent):
    # select sentence
    sent = list(doc.sents)[num_sent]
    # print constituent parse string
    print(sent._.parse_string)

lookup = {u'decide': u'decide',
         u'decides': u'decide',
         u'decided': u'decide',
         u'deciding': u'decide',
         u'check': u'check',
         u'checks': u'check',
         u'checked': u'check',
         u'checking': u'check',
         u'determine': u'determine',
         u'determines': u'determine',
         u'determined': u'determine',
         u'determining': u'determine',
         u'investigate': u'investigate',
         u'investigates': u'investigate',
         u'investigated': u'investigate',
         u'investigating': u'investigate'}

from spacy.lemmatizer import Lemmatizer
lemmatizer = Lemmatizer(lookup=lookup)

def is_not_special_case(subtree):

  if subtree.parent().label() == "VP" and subtree.leaves()[0] in ["whether", "where", "if", "which", "what"]:
    if lemmatizer.lookup(subtree.parent().leaves()[0]) in ["check", "decide", "investigate", "determine"]:
      return False

  return True

def create_tree(sent) :
    # create a tree from the parsed sentence (string)
    t = ParentedTree.fromstring(sent._.parse_string)

    # create empty list for subsentences
    subtexts = []
    subtexts.append(' '.join(t.leaves()))
    # look for subsentences and append them to the list
    for subtree in t.subtrees() :
        if subtree.label() == "SBAR" and not is_not_special_case(subtree):
          return subtexts
        else :
          if subtree.label()=="SBAR" and is_not_special_case(subtree) :
            subtexts.append(' '.join(subtree.leaves()))
          if subtree.label()=="S" and subtree.parent() is not None:
            if subtree.parent().label() in ["PP","S","ADVP","VP","NP"] :
              subtexts.append(' '.join(subtree.leaves()))
    return subtexts

def string_diff(string1, string2) :
    return string1.replace(string2,"")

def isSubstring(sub,sent) :
    return sent.lower().replace(' ','').find(sub.lower().replace(' ','')) >= 0

def remove_double_subsents(subs) :
    temp = []
    final = []
    # check all subsentences is list except first one, first is already compared with second
    while len(subs) > 0:
        sub_last = subs[-1] # take last subsentence
        # compare with all subsentences except last one
        for sub2 in subs[0:-1] :
            if isSubstring(sub_last,sub2) : # check if last substring appears in other substring
                temp.append(string_diff(sub2,sub_last)) # yes: append substring with last string removed
            else :
                temp.append(sub2) # no: append substring
        final.insert(0,sub_last.strip()) # insert last string because it is checked now
        subs = temp # reprocess but with already shorter subs now
        temp = []

    return final

def remove_double_subsents2(subtexts) :
    subsent = []
    for i in range(len(subtexts)) :
        if i == len(subtexts)-1 :
            subsent.append(subtexts[i])
        else :
            subsent.append(string_diff(subtexts[i], subtexts[i+1]))

    return subsent

def capitalize_first_letters(subtexts) :
    subsent = []
    for s in subtexts :
        index = 0
        for char in s :
            if char.isalnum() == False :
                index = index +1
            else :
                subsent.append(s[index].upper() + s[index+1:])
                break
    return subsent

# Append certain words to the next subsentence
def concatenate_sep_words(subtexts) :
    sep_words = ["in order","which","once","as","if","when","who", "until","if not",
                 "whether","that","whenever","again","for","while","before ","for which","and"]
    i = 0
    for i in range(len(subtexts)) :
        if subtexts[i] in sep_words and i < len(subtexts) - 1:
            subtexts[i+1] = subtexts[i] + " " + subtexts[i+1]
    return remove_single_sep_words(subtexts, sep_words) #remove single seperate words from list

def remove_single_sep_words(subtexts, sep_words) :
    return ([w for w in subtexts if w.lower().strip() not in sep_words])

def reorder_subsents(sent, subtexts) :
    order = {}
    minimum = 0
    maximum = 10
    for s in subtexts :
        order.update({s : sent.lower().replace(' ','').find(s[0:15].lower().replace(' ','')) })
    return sorted(order, key=order.__getitem__)

def lower_no_punct(subtexts) :
    exclude = (set(string.punctuation))
    #remove punctuation
    subtexts = [''.join(ch for ch in s if ch not in exclude) for s in subtexts]
    #replace multiple whitespaces by one, exactly one space between every word.
    return [' '.join(s.lower().split()) for s in subtexts]
