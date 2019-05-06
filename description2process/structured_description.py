import pandas as pd
import spacy
import re
import string

# -- public functions
def get_structured_description(description, labeled_clauses):
    clauses = labeled_clauses.loc[labeled_clauses['label'] == 1]
    clauses = clauses[['clause','activity']]

    description = description.translate(str.maketrans('', '', string.punctuation))

    for index, row in clauses.iterrows() :
        clause = row['clause'].lower().strip()
        conj = get_conjunction(clause)
        act = conj + " <act> " + str(row['activity']) + " </act> "
        # if the clause is found back exactly in the description, replace it by its activity
        pattern = re.compile(re.escape(clause), re.IGNORECASE)
        if pattern.search(description):
          description = pattern.sub(act, description)
        # replace all charachters between the first 10 characters of the clause and
        # the last 10 character by the corresponding activity
        else :
          pattern2 = re.compile(str(clause[0:10]) + ".*?" + str(clause[-10:]), re.IGNORECASE)
          description = pattern2.sub(act,description)

    return description

def get_structured_description_df(descriptions, labeled_clauses):
    descriptions['structured_description'] = descriptions.apply(lambda row: get_structured_description_row(row, labeled_clauses), axis = 1)
    return descriptions

# -- Private functions
conjunctions = ["first","the process starts when","initially","subsequently","then","afterwards,","later","thereafter","after that","in the next step","when","after","once","before", "next"]
def get_conjunction(clause):
    for c in conjunctions:
        if c in clause:
          return c
    # No conjunction found: retun empty string
    return ""

def get_structured_description_row(description_row, labeled_clauses) :

    clauses = labeled_clauses.loc[labeled_clauses['label'] == 1]
    clauses = clauses.loc[labeled_clauses['id_description'] == description_row['id_description']]
    clauses = clauses[['clause','activity']]

    description = description_row['description'].translate(str.maketrans('', '', string.punctuation))

    for index, row in clauses.iterrows() :
        clause = row['clause'].lower().strip()
        conj = get_conjunction(clause)
        act = conj + " <act> " + str(row['activity']) + " </act> "
        # if the clause is found back exactly in the description, replace it by its activity
        pattern = re.compile(re.escape(clause), re.IGNORECASE)
        if pattern.search(description):
          description = pattern.sub(act, description)
        # replace all charachters between the first 10 characters of the clause and
        # the last 10 character by the corresponding activity
        else :
          pattern2 = re.compile(str(clause[0:10]) + ".*" + str(clause[-10:]), re.IGNORECASE)
          description = pattern2.sub(act,description)

    return description
