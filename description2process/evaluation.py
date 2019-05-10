import xml.etree.ElementTree as ET
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from description2process import xml_model

# -- Functions for the evaluation of activities
def get_activities(tree, activities):
  for child in tree:
    activities.append(child.text)

def get_all_activities(tree):
  activities = list()
  for child in tree:
    if child.tag != "act":
        get_activities(child, activities)
    else:
      activities.append(child.text)

  return activities

def get_score_activity_length(list_solution, list_prediction):
  length_solution = len(list_solution)
  length_prediction = len(list_prediction)

  maximum = np.max([length_solution, length_prediction])
  if maximum  == 0:
    return 1
  else :
    score_length = 1 - (abs(length_solution - length_prediction)/ maximum )
    return score_length

# -- Functions for the evaluation of the tags
def get_tags(tree, tags, tag):
  for child in tree :
    tags.append(child.tag)
  tags.append(tag + "_close")

def get_all_tags(tree):
  tags = list()
  for child in tree:
    if child.tag in ["path1","path2","path3"] :
      tags.append(child.tag)
      get_tags(child, tags, child.tag)
    else :
      tags.append(child.tag)

  return tags

def get_blue_score_tags(list_prediction, list_solution) :
  return sentence_bleu([list_solution], list_prediction, weights = (0,1,0,0))

# -- Functions for the evaluation of the branches
def get_number_of_branches(tree) :
  n_branches = 0
  for child in tree :
    if child.tag in ["path1","path2","path3"]:
      n_branches = n_branches + 1
  return n_branches

def get_score_branches(n_branches_prediction, n_branches_solution) :
  maximum = np.max([n_branches_solution, n_branches_prediction])
  if maximum == 0 :
    return 1
  else :
    score_branches =  1- ( (abs(n_branches_solution - n_branches_prediction)) / maximum )
    return score_branches

# -- Functions for order activities
def is_same_activity(act1, act2):
  list_act1 = act1.lower().split()
  list_act2 = act2.lower().split()
  length_list_act1 = len(list_act1)
  length_list_act2 = len(list_act2)
  if length_list_act1 > length_list_act2:
    long = list_act1
    short = list_act2
  else :
    long = list_act2
    short = list_act1

  count = 0
  for word in short :
    if word in long:
      count = count + 1

  score = count / len(short)
  if score > 0.5:
    return True
  else :
    return False

def get_score_order_activities(activities_prediction, activities_solution):
  # create dictionary from activites prediction
  list_act = list()
  for i in range(len(activities_prediction)):
    list_act.append("act" + str(i))
  dict_pred = dict(zip(activities_prediction, list_act))

  #Replace activities in solution with short form of prediction
  for key, value in dict_pred.items():
    # Find an exact match
    if key in activities_solution:
      activities_solution[activities_solution.index(key)] = value
    else :
    # Find match on similarity
      for act_sol in activities_solution :
        if is_same_activity(key, act_sol):
          activities_solution[activities_solution.index(act_sol)] = value

  # Replace activities that were not found by 'no_act' in solution
  for a in activities_solution:
    if a not in list_act:
      activities_solution[activities_solution.index(a)] = "no_act"

  # Compute and return BLEU score between both lists
  return sentence_bleu([activities_solution], list_act, weights = (0,1,0,0))

def get_well_formed_xml(xml):
  list_pred = xml.split()
  i = 0

  tag_act = False
  tag_path1 = False
  tag_path2 = False
  tag_path3 = False
  while i < len(list_pred) :
    word = list_pred[i]

    # check if all activities are closed when new tag is opened or path closed
    if word in ['<act>','<path1>','<path2>','<path3>', '</path1>','</path2>','</path3>'] and tag_act :
      list_pred.insert(i, '</act>')
      tag_act = False
      i = i+1

    # check if all path1 is closed before opening new path
    if word in ['<path1>','<path2>','<path3>'] and tag_path1 :
      list_pred.insert(i, '</path1>')
      tag_path1 = False
      i = i+1

    if word in ['<path1>','<path2>','<path3>'] and tag_path2 :
      list_pred.insert(i, '</path2>')
      tag_path2 = False
      i = i+1

    if word in ['<path1>','<path2>','<path3>'] and tag_path3 :
      list_pred.insert(i, '</path3>')
      tag_path3 = False
      i = i+1

    # Navigate xml string
    if word == '<act>':
      tag_act = True
    elif word == '</act>':
      tag_act = False
    elif word == '<path1>':
      tag_path1 = True
    elif word == '</path1>':
      tag_path1 = False
    elif word == '<path2>':
      tag_path2 = True
    elif word == '</path2>':
      tag_path2 = False
    elif word == '<path3>':
      tag_path3 = True
    elif word == '</path3>':
      tag_path3 = False
    else :
      pass

    i = i+1

  # Check if all tags are properly closed
  if tag_act or tag_path1 or tag_path2 or tag_path3:
    if tag_act :
      list_pred.append('</act>')
      tag_act = False

    if tag_path1 :
      list_pred.append('</path1>')
      tag_path1 = False

    if tag_path2 :
      list_pred.append('</path2>')
      tag_path2 = False

    if tag_path3 :
      list_pred.append('</path3>')
      tag_path3 = False

  return ' '.join(list_pred)
  
# -- Main fucntion
#-----------------

# -- Complete evaluation
def get_evaluation(prediction, solution):
  try:
    tree_prediction = ET.fromstring("""<?xml version="1.0"?> <data> """ + prediction.strip() + " </data>")
  except:
    prediction = get_well_formed_xml(prediction)
    tree_prediction = ET.fromstring("""<?xml version="1.0"?> <data> """ + prediction.strip() + " </data>")

  tree_solution = ET.fromstring("""<?xml version="1.0"?> <data> """ + solution.strip() + " </data>")

  activities_prediction = get_all_activities(tree_prediction)
  activities_solution = get_all_activities(tree_solution)

  tags_prediction = get_all_tags(tree_prediction)
  tags_solution = get_all_tags(tree_solution)

  n_branches_prediction = get_number_of_branches(tree_prediction)
  n_branches_solution = get_number_of_branches(tree_solution)

  score_activities = get_score_activity_length(activities_prediction, activities_solution)
  score_branches = get_score_branches(n_branches_prediction, n_branches_solution)
  score_bleu_tags = get_blue_score_tags(tags_prediction, tags_solution)
  score_bleu_act = get_score_order_activities(activities_prediction, activities_solution)

  return score_activities, score_branches, score_bleu_tags, score_bleu_act
