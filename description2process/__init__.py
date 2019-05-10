import tensorflow as tf

# We need to enable eager execution for inference at the end of this notebook.
tfe = tf.contrib.eager
tfe.enable_eager_execution()

TFVERSION='1.13'
import os
os.environ['TFVERSION'] = TFVERSION


# Import library
from description2process import data_generation
from description2process import contraction_expansion
from description2process import coreference_resolution
from description2process import clause_extraction
from description2process import activity_recognition
from description2process import activity_extraction
from description2process import structured_description
from description2process import xml_model
from description2process import visualization
from description2process import evaluation

# Returns the visualisation of a process description
# INPUT: process description in string format
def description2model(description, png = False):
    # step1 : contraction expansion
    description = contraction_expansion.expand_contractions(description)
    print("Step 1/8 DONE: contraction expansion")
    # step2 : coreference resolution
    description = coreference_resolution.resolve_coreferences(description)
    print("Step 2/8 DONE: coreference resolution")
    # step3 : clause extraction
    subsentences = clause_extraction.get_clauses(description)
    print("Step 3/8 DONE: extracted clauses ")
    # step4: label clauses
    labeled_clauses_df = activity_recognition.contains_activity_list(subsentences)
    print("Step 4/8 DONE: labeled clauses ")
    # step5: activity extraction
    df_activities = activity_extraction.get_activity_df(labeled_clauses_df)
    print("Step 5/8 DONE: extracted activities ")
    # step6: get a structured_descriptions
    str_descr = structured_description.get_structured_description(description, df_activities)
    print("Step 6/8 DONE: semi-structured descriptions")
    # step7: get XML format of models
    xml = xml_model.structured2xml(str_descr)
    print("Step 7/8 DONE: model in XML")
    # step8: Visualize the model in xml
    model = visualization.xml2model(xml, png)
    print("Step 8/8 DONE: Visualize model")

    return model

# Returns the xml format of the process description
# INPUT: process description in string format
def description2xml(description):
    # step1 : contraction expansion
    description = contraction_expansion.expand_contractions(description)
    print("Step 1/7 DONE: contraction expansion")
    # step2 : coreference resolution
    description = coreference_resolution.resolve_coreferences(description)
    print("Step 2/7 DONE: coreference resolution")
    # step3 : clause extraction
    subsentences = clause_extraction.get_clauses(description)
    print("Step 3/7 DONE: extracted clauses ")
    # step4: label clauses
    labeled_clauses_df = activity_recognition.contains_activity_list(subsentences)
    print("Step 4/7 DONE: labeled clauses ")
    # step5: activity extraction
    df_activities = activity_extraction.get_activity_df(labeled_clauses_df)
    print("Step 5/7 DONE: extracted activities ")
    # step6: get a structured_descriptions
    str_descr = structured_description.get_structured_description(description, df_activities)
    print("Step 6/7 DONE: semi-structured descriptions")
    # step7: get XML format of models
    xml = xml_model.structured2xml(str_descr)
    print("Step 7/7 DONE: model in XML")

    return xml

# returns the structured description of raw process descriptions
# Input: pandas dataframe of process descriptions
def description2structured_df(description_df):
    # step1 : contraction expansion
    description_df = contraction_expansion.expand_contractions_df(description_df)
    print("Step 1/6 DONE: contraction expansion")
    # step2 : coreference resolution
    description_df = coreference_resolution.resolve_coreferences_df(description_df)
    print("Step 2/6 DONE: coreference resolution")
    # step3 : clause extraction
    description_df = clause_extraction.get_clauses_df(description_df)
    print("Step 3/6 DONE: extracted clauses ")
    # step4: label clauses
    labeled_clauses = activity_recognition.contains_activity_df(description_df)
    print("Step 4/6 DONE: labeled clauses ")
    # step5: activity extraction
    df_activities = activity_extraction.get_activity_df(labeled_clauses)
    print("Step 5/6 DONE: extracted activities ")
    # step6: get a structured_descriptions
    str_descr = structured_description.get_structured_description_df(description_df, df_activities)
    print("Step 6/6 DONE: returned structured descriptions")

    return str_descr

# return the descripition after contraction expansion and coreference resolution.
# This type of description can be seen as a cleaned version of the original one.
# Input: pandas dataframe of process descriptions
def description2referenceresolved_df(description_df):
    # step1 : contraction expansion
    description_df = contraction_expansion.expand_contractions_df(description_df)
    # step2 : coreference resolution
    description_df = coreference_resolution.resolve_coreferences_df(description_df)

    return description_df

# Return the description with a list containing the description's extracted clauses
# Input: pandas dataframe of process description
def description2clauses_df(description_df):
    # step1 : contraction expansion
    description_df = contraction_expansion.expand_contractions_df(description_df)
    # step2 : coreference resolution
    description_df = coreference_resolution.resolve_coreferences_df(description_df)
    # step3 : clause extraction
    description_df = clause_extraction.get_clauses_df(description_df)

    return description_df

# Return the description with a list containg the descriptions's extracted clauses
# + an extra dataframe with all its labeled clauses
# Input: pandas dataframe of process descriptions
def description2labeledclauses_df(description_df):
    # step1 : contraction expansion
    description_df = contraction_expansion.expand_contractions_df(description_df)
    # step2 : coreference resolution
    description_df = coreference_resolution.resolve_coreferences_df(description_df)
    # step3 : clause extraction
    description_df = clause_extraction.get_clauses_df(description_df)
    # step4: label clauses
    labeled_clauses = activity_recognition.contains_activity_df(description_df)

    return labeled_clauses, description_df
