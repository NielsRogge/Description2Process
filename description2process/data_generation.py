import random
import pandas as pd
import numpy as np
import os, sys

def determine_path ():
    """Borrowed from wxglade.py"""
    try:
        root = __file__
        if os.path.islink (root):
            root = os.path.realpath (root)
        return os.path.dirname (os.path.abspath (root))
    except:
        print ("I'm sorry, but something is wrong.")
        print ("There is no __file__ variable. Please contact the author.")
        sys.exit ()

def check_datafiles_in_package ():
    print ("module is running")
    print (determine_path ())
    print ("My various data files are:")
    files = [f for f in os.listdir(determine_path () + "/data")]
    print (files)

def read_data_generation_content():
    data = pd.read_excel(determine_path () + "/data/data_generation_content.xlsx")
    return data

data = pd.read_excel(determine_path () + "/data/data_generation_content.xlsx")

begin = ["first","the process starts when","initially"]
info = ["next","subsequently"]
body = ["next,","next","then","afterwards,","later","thereafter","","", "subsequently","after that,","in the next step,"]
double = ["when","after","once"]
replace = ["before"]
condition = ['If ',"If ","If ","In case ","Whenever "]

def get_list_scenarios():
    return data.scenario.unique()

def get_description():
    scenario = random.choice(get_list_scenarios())
    all_ordered_clauses = get_scenario_content(scenario)
    text, solution = create_text(all_ordered_clauses)
    return text, solution.strip()

def get_descriptions(start =0, nscenarios = 10, ndescriptions = 3, data_frame = False) :
    list_texts = []
    list_solutions = []

    # Get list of all scenarios
    scenarios = get_list_scenarios()
    for scenario in scenarios[start:start+nscenarios] :

        for _ in range(ndescriptions) :
            all_ordered_clauses = get_scenario_content(scenario)
            text, solution = create_text(all_ordered_clauses)
            list_texts.append(text)
            list_solutions.append(solution)

    if data_frame :
        df_descriptions = pd.DataFrame({"description": list_texts,"solution":list_solutions})
        df_descriptions['id_description'] = df_descriptions.index
        return df_descriptions
    else :
        return list_texts, list_solutions


def get_scenario_content(scenario) :
  # Select all data for certain scenario
  scenario_df = data.loc[data['scenario'] == scenario]
  scenario_df = scenario_df.replace(np.nan, '', regex=True)

  # get all ordered clauses
  all_ordered_clauses = []
  max_steps = int(max(scenario_df['type2']))

  for i in range(1,max_steps+1):
    block_temp = scenario_df.loc[scenario_df["type2"] == i][["type1","ConjunctionNeeded","sentence","solution"]]
    all_ordered_clauses.append(block_temp)

  return all_ordered_clauses

def get_first_sentence (df_first) :
    # Select a random row from all the options
    random_row = df_first.sample(n=1)

    # Check if conjunction is needed
    if is_conjunction_needed(random_row) :
      text = random.choice(begin) + " " + random_row.iat[0,2] + "."
      solution = random_row.iat[0,3]
    else :
      text = random_row.iat[0,2] + "."
      solution = random_row.iat[0,3]

    return text.strip().capitalize(), solution

def get_general_info(random_row) :
    # Select a random row from all the options
    #random_row = df_info.sample(n=1)

    # Check if conjunction is needed
    if is_conjunction_needed(random_row) :
      text = random.choice(info) + " " + random_row.iat[0,2] + "."
      solution = random_row.iat[0,3]
    else :
      text = random_row.iat[0,2] + "."
      solution = random_row.iat[0,3]

    return text.strip().capitalize(), solution

def get_body_sentence(random_row) :
    # Select a random row from all the options
    #random_row = df_body.sample(n=1)

    # Check if conjunction is needed
    if is_conjunction_needed(random_row) :
      text = random.choice(body) + " " + random_row.iat[0,2] + "."
      solution = random_row.iat[0,3]
    else :
      text = random_row.iat[0,2] + "."
      solution = random_row.iat[0,3]

    return text.strip().capitalize(), solution

def get_double_sentence(random_row1,random_row2) :
    # Select a random row from all the options for each df
    #random_row1 = df_clause1.sample(n=1)
    #random_row2 = df_clause2.sample(n=1)

    text = random.choice(double) + " " + random_row1.iat[0,2] + ", " + random_row2.iat[0,2] + "."
    solution = random_row1.iat[0,3] + " " + random_row2.iat[0,3]

    return text.strip().capitalize(), solution

def get_replace_sentence(random_row1,random_row2) :
    # Select a random row from all the options for each df
    #random_row1 = df_clause1.sample(n=1)
    #random_row2 = df_clause2.sample(n=1)

    text = random.choice(replace) + " " + random_row2.iat[0,2] + ", " + random_row1.iat[0,2] + "."
    solution = random_row1.iat[0,3] + " " + random_row2.iat[0,3]

    return text.strip().capitalize(), solution

def get_split_path1(df_clause) :
  condition_path1 = df_clause.loc[df_clause["type1"] == "condition_path1"]
  result_path1 = df_clause.loc[df_clause["type1"] == "condition_result_path1"]

  conjunction1 = ""
  if is_conjunction_needed(condition_path1) :
    conjunction1 = random.choice(condition)

  text = conjunction1 + condition_path1.iat[0,2]+ ", " + result_path1.iat[0,2] + "."

  solution = "<path1>" + result_path1.iat[0,3] + "</path1>"

  return text.strip().capitalize(), solution


def get_split_path2(df_clause) :
  condition_path1 = df_clause.loc[df_clause["type1"] == "condition_path1"]
  condition_path2 =  df_clause.loc[df_clause["type1"] == "condition_path2"]
  result_path1 = df_clause.loc[df_clause["type1"] == "condition_result_path1"]
  result_path2 = df_clause.loc[df_clause["type1"] == "condition_result_path2"]

  conjunction1 = ""
  conjunction2 = ""
  if is_conjunction_needed(condition_path1) :
    conjunction1 = random.choice(condition)
  if is_conjunction_needed(condition_path2) :
    conjunction2 = random.choice(condition)

  text = conjunction1.capitalize() + condition_path1.iat[0,2]+ ", " + result_path1.iat[0,2] + ". " + conjunction2.capitalize() + condition_path2.iat[0,2] + ", " + result_path2.iat[0,2] + "."

  solution = "<path1>" + result_path1.iat[0,3] + "</path1>" + "<path2>" + result_path2.iat[0,3] + "</path2>"

  return text.strip(), solution


def get_split_path3(df_clause) :
  condition_path1 = df_clause.loc[df_clause["type1"] == "condition_path1"]
  condition_path2 =  df_clause.loc[df_clause["type1"] == "condition_path2"]
  condition_path3 =  df_clause.loc[df_clause["type1"] == "condition_path3"]
  result_path1 = df_clause.loc[df_clause["type1"] == "condition_result_path1"]
  result_path2 = df_clause.loc[df_clause["type1"] == "condition_result_path2"]
  result_path3 = df_clause.loc[df_clause["type1"] == "condition_result_path3"]

  conjunction1 = ""
  conjunction2 = ""
  conjunction3 = ""
  if is_conjunction_needed(condition_path1) :
    conjunction1 = random.choice(condition)
  if is_conjunction_needed(condition_path2) :
    conjunction2 = random.choice(condition)
  if is_conjunction_needed(condition_path3) :
    conjunction3 = random.choice(condition)

  text = conjunction1.capitalize() + condition_path1.iat[0,2]+ ", " + result_path1.iat[0,2] + ". " + conjunction2.capitalize() + condition_path2.iat[0,2] + ", " + result_path2.iat[0,2] + ". " + conjunction3.capitalize()  + condition_path3.iat[0,2] + ", " + result_path3.iat[0,2] + "."

  solution = "<path1>" + result_path1.iat[0,3] + "</path1>" + "<path2>" + result_path2.iat[0,3] + "</path2>" + "<path3>" + result_path3.iat[0,3] + "</path3>"

  return text.strip(), solution

def get_type_and_prepare_block(df_clauses):
  random_row = df_clauses.sample(n=1)
  type1 = random_row.iat[0,0].strip()

  if type1 == "step" :
    return "step", random_row

  elif type1 == "general_info" :
    return "general_info", random_row

  elif type1 in ["condition_path1","condition_path2","condition_path3","condition_result_path1","condition_result_path2","condition_result_path3"] :
    all_types = df_clauses.iloc[:,0]
    # path1, path2, path3
    if all_types.str.contains('condtion_path3', regex = False).any() :
      random_path1 = df_clauses.loc[df_clauses["type1"] == "condition_path1"].sample(n=1)
      random_result_path1 = df_clauses.loc[df_clauses["type1"] == "condition_result_path1"].sample(n=1)
      random_path2 = df_clauses.loc[df_clauses["type1"] == "condition_path2"].sample(n=1)
      random_result_path2 = df_clauses.loc[df_clauses["type1"] == "condition_result_path2"].sample(n=1)
      random_path3 = df_clauses.loc[df_clauses["type1"] == "condition_path3"].sample(n=1)
      random_result_path3 = df_clauses.loc[df_clauses["type1"] == "condition_result_path3"].sample(n=1)
      random_df = pd.concat([random_path1, random_path2, random_path3, random_result_path1, random_result_path2, random_result_path3])
      return "split_path3", random_df

    # path1 and path2
    if all_types.str.contains('condition_path2', regex=False).any() :
      random_path1 = df_clauses.loc[df_clauses["type1"] == "condition_path1"].sample(n=1)
      random_result_path1 = df_clauses.loc[df_clauses["type1"] == "condition_result_path1"].sample(n=1)
      random_path2 = df_clauses.loc[df_clauses["type1"] == "condition_path2"].sample(n=1)
      random_result_path2 = df_clauses.loc[df_clauses["type1"] == "condition_result_path2"].sample(n=1)
      random_df = pd.concat([random_path1, random_path2, random_result_path1, random_result_path2])
      return "split_path2", random_df

    # only path 1
    if all_types.str.contains('condition_path1', regex = False).any() :
      random_path1 = df_clauses.loc[df_clauses["type1"] == "condition_path1"].sample(n=1)
      random_result_path1 = df_clauses.loc[df_clauses["type1"] == "condition_result_path1"].sample(n=1)
      random_df = pd.concat([random_path1, random_result_path1])
      return "split_path1", random_df
    else:
      return "ERROR, path not find", df_clauses

  else :
    return "ERROR: somenthing went wrong with the types!", df_clauses

def get_sentence_with_multiple_steps(df_clause1, df_clause2):
  # Choose a random structure to be used in the text
  # "double" and "replace" structure can only be used if there are at least 2 steps over
  if is_conjunction_needed(df_clause1) and is_conjunction_needed(df_clause2) :
    type_sent = random.choice(["body","double","replace","double","replace"])
  else :
    type_sent = "body"

  if type_sent == "body" :
    # df_clause1
    text_temp1, solution_temp1 = get_body_sentence(df_clause1)

    # df_clause2
    text_temp2, solution_temp2 = get_body_sentence(df_clause2)

    text_temp = text_temp1 + " " + text_temp2
    solution_temp = solution_temp1 + " " + solution_temp2

  elif type_sent == "double" :
    text_temp, solution_temp = get_double_sentence(df_clause1, df_clause2)

  elif type_sent == "replace" :
    text_temp, solution_temp = get_replace_sentence(df_clause1, df_clause2)

  else :
    text_temp = " ERROR: in choice type of sentence."
    solution_temp = " ERROR: in choice type of sentence."

  return text_temp, solution_temp

def is_conjunction_needed(row):
  return row.iat[0,1] == 1

def is_next_step_of_type_step(next_step) :
  if next_step.iat[0,0] == "step":
    return True
  else :
    return False

def create_text(all_ordered_clauses) :

    # initialize text and solution
    text = ""
    solution = ""
    is_first_step = True

    # Loop through all ordered clauses
    while len(all_ordered_clauses) > 0 :

        # Check what the next block looks like: is it a step, a condition, general_info?
        # and randomly select the clauses that will be used for this block
        type1, df_clause = get_type_and_prepare_block(all_ordered_clauses.pop(0))

        # if statement for every type
        if type1 == "step" :

            # Add first sentence to text (only done once)
            if is_first_step :
              text_temp, solution_temp = get_first_sentence(df_clause)
              is_first_step = False

            else :
              # Check if there are 2 consecutive blocks of the type "step"
              if ( len(all_ordered_clauses) > 0) and (is_next_step_of_type_step(all_ordered_clauses[0]) ) :
                # Two blocks of the type "step", get randomly a structure
                text_temp, solution_temp = get_sentence_with_multiple_steps(df_clause, all_ordered_clauses.pop(0))

              else :
                # Only one block has type "step", so only one option for sentence construction
                text_temp, solution_temp = get_body_sentence(df_clause)

        elif type1 == "split_path1" :
          text_temp, solution_temp = get_split_path1(df_clause)

        elif type1 == "split_path2" :
          text_temp, solution_temp = get_split_path2(df_clause)

        elif type1 == "split_path3" :
          text_temp, solution_temp = get_split_path3(df_clause)

        elif type1 == "general_info" :
          text_temp, solution_temp = get_general_info(df_clause)

        else :
           text = text + " ERROR: in choice type of sentence."

        # add new sentence and solution to full description and complete solution
        text = text + " " + text_temp
        solution = solution + " " + solution_temp

    return text.strip(), solution
