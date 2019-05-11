from graphviz import Digraph
import xml.etree.ElementTree as ET
import pandas as pd
import pydot
import os

## -- Internal functions -- #

# -- Functions for the creation of graphical objects -- #
def initialise_graph():
  dot = Digraph(comment="Business Process Model")
  dot.attr('graph', rankdir = "LR")
  return dot

def create_begin(dot, connector):
  dot.attr('node', shape="circle")
  dot.attr('node')
  dot.node("begin")
  connector.append("begin")
  return dot, connector

def create_end(dot, connector):
  dot.attr('node', shape="circle")
  dot.attr('node')
  dot.node("end")
  create_edge(connector.pop(),"end", dot)
  return dot, connector

def create_activity(act, dot) :
    dot.attr('node', shape ="box")
    dot.attr('node')
    dot.node(act)
    return dot

def create_edge(act1,act2,dot) :
    dot.edge(act1,act2)
    return dot

# -- Recursive function to navigate xml tree
def loop_level_tree(tree, dot, connector):
  after_else = False
  for child in tree :
    if child.tag == "act" and not after_else :
      create_activity(child.text, dot)
      create_edge(connector.pop(),child.text, dot)
      connector.append(child.text)

    if child.tag == "act" and after_else :
      create_activity(child.text, dot)
      create_edge(connector.pop(),child.text, dot)
      create_edge(connector.pop(),child.text, dot)
      connector.append(child.text)
      after_else = False

    if child.tag == "path1" :
      connector.append(connector[-1])
      loop_level_tree(child, dot, connector)

    if child.tag == "path2" :
      connector[-2], connector[-1] = connector[-1], connector[-2]
      loop_level_tree(child, dot, connector)
      after_else = True

  return dot

## -- Main functions -- ##

def xml2model(xml, png = False):
    process = """<?xml version="1.0"?> <data> """ + xml + " </data>"
    # get tree from XML
    tree = ET.fromstring(process)

    # variables to keep track of process
    connector = []
    after_else = False

    # Initialise graph
    dot = initialise_graph()
    # create begin node
    dot, connector = create_begin(dot, connector)
    # Loop throug XML file
    dot = loop_level_tree(tree, dot, connector)
    # create begin node
    dot = create_end(dot, connector)

    if png:
        dot[0].render(filename='process_model.dot')
        (graph,) = pydot.graph_from_dot_file('process_model.dot')
        graph.write_png('process_model.png')
        os.remove("process_model.dot")

    return dot[0]
