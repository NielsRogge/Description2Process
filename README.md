# Description2Process
This repository includes all data as well as source code of the paper "On the Applicability of Deep Learning to construct Process Models from Natural Text", written by Simeon Devos and Niels Rogge (2019). Besides that, there is an interactive notebook that allows the user to use and test all functionality at once. This notebook can be found in the 'description2process' directory. 

The proposed methodology consists of an **NLP pipeline** that transforms a textual description into a process model in an XML format, which eventually can be mapped to an image that is roughly based on BPMN. The NLP pipeline consists of 8 sequential steps (of which steps 2, 3, 4, 5 and 7 use deep learning):

1. Contraction expansion
2. Coreference resolution
3. Clause extraction
4. Activity recognition
5. Activity extraction
6. Construction of semi-structured description
7. Transformation of semi-structured description to XML format
8. Visualization of XML format 

Textual inputs are limited to sequences of activities and 2 types of XOR splits, as described in the paper.

## Installation 
The project requires Python 3. First, 4 libraries need to be installed that do not (yet) install automatically with the 'description2process' library: 
```
pip install spacy==2.1.3
pip install neuralcoref==4.0
pip install benepar
pip install allennlp
```

Next, one can install the library as follows:
```
pip install description2process
```

## Usage 
All steps of the pipeline are available as separate Python modules which one can test and use. Note that each step takes as input the output of the previous step, so one should execute the different steps sequentially. 

```
import description2process as d2p
from description2process import contraction_expansion
from description2process import coreference_resolution
from description2process import clause_extraction
from description2process import activity_recognition
from description2process import activity_extraction
from description2process import structured_description
from description2process import xml_model
from description2process import visualization
```
### Create description
To start, first enter a textual description yourself: 
```
description = "First the secretary clarifies the shipment method. Whenever there's a requirement for large amounts, he selects one of three logistic companies. Whenever small amounts are required, a package label is written by the secretary. Subsequently the goods can be packaged by the warehousemen. Next, the packaged goods are prepared for being picked up by the logistic company."
```
If one wants to compare the solution generated by our system with a reference solution in the end, one should store the corresponding reference solution (in XML format) in a variable as follows:
```
solution = "<act> clarify shipment method </act> <path1> <act> select one of three logistic companies </act> </path1> <path2> <act> write package label </act> </path2> <act> package goods </act> <act> prepare packaged goods </act> <act> pick up </act>"
```
Alternatively, one can let our data generation algorithm create a random description, together with its reference solution, based on the 120 scenarios as mentioned in the paper. The data generation algorithm is available as a separate module in the library, as one can see below:
```
from description2process import data_generation
description, solution = d2p.data_generation.get_description()
print(description)
print(solution)
```

### Contraction expansion 
The first step of the pipeline expands all contractions of the description, like "doesn't" and "isn't" to "does not" and "is not". Of course, if the description does not contain any contractions, one can skip this step. Contraction expansion can be done as follows:
```
description = d2p.contraction_expansion.expand_contractions(description)
print(description)
```

### Coreference resolution 
Subsequently, coreference resolution can be performed. As mentioned in the paper, only coreferences are resolved for the words "he", "she" and "it". This can be done as follows:
```
description = d2p.coreference_resolution.resolve_coreferences(description)
print(description)
```

### Clause extraction
To split up a given description into a list of separate clauses, the clause extraction algorithm can be called as follows:
```
clauses = d2p.clause_extraction.get_clauses(description)
print(clauses)
```

### Activity recognition
Next, activity recognition can be performed on the extracted clauses. This involves a deep neural network classifying each clause separately. The result is a Pandas dataframe with labeled clauses, where "True" indicates that the clause includes an activity, whereas "False" indicates that the clause does not include an activity. Activity recognition can be done as follows:
```
labeled_clauses_df = d2p.activity_recognition.contains_activity_list(clauses)
print(labeled_clauses_df)
```

### Activity extraction
To extract the activities of the clauses that contain an activity, the activity extraction algorithm can be called. This simply adds a column to the Pandas dataframe obtained in the previous step that includes the extracted activities. For clauses that do not contain an activity, "NaN" is displayed. The activity extraction algorithm can be called as follows:
```
extracted_activities_df = d2p.activity_extraction.get_activity_df(labeled_clauses_df)
print(extracted_activities_df)
```  

### Construction of semi-structured description
Next, one can construct a semi-structured description based on the original description, cleaned during the contraction expansion and coreference resolution steps. This step replaces the clauses that contained an activity (obtained in the activity recognition step) by their extracted activities (obtained in the activity extraction step), surrounded by \<act> and \</act> tags. This can be done as follows:
```
structured_description = d2p.structured_description.get_structured_description(description, extracted_activities_df)
print(structured_description)
```

### Transformation of semi-structured description to XML format
Next, a Transformer model can be called to translate the semi-structured description to a complete XML format that represents the process model. If appropriate, the model should add <path> and </path> tags to passages describing splits and merges. The Transformer model can be called as shown below. Currently, this function also prints log information regarding the Transformer model. 
```
xml = d2p.xml_model.structured2xml(structured_description)
print(xml)
```

### Visualization of generated XML format 
Finally, the generated XML format can be mapped to an image (graph) as follows:
```
image = d2p.visualization.xml2model(xml)
image
```
In case a reference solution was specified, one can also visualize it for comparison with the generated solution:
```
image = d2p.visualization.xml2model(solution)
image
```

### Evaluation of generated XML format 
To compare the generated XML format to the reference solution, one can compute the 4 performance metrics mentioned in the paper. The evaluation is available as a separate module, as shown below: 
```
from description2process import evaluation
score_activities, score_branches, score_bleu_tags, score_bleu_act = d2p.evaluation.get_evaluation(xml, solution)

print("Score number of activities: {0:.0%}".format( round( score_activities, 4) ) ) 
print("Score number of branches: {0:.0%}".format( round( score_branches, 4 ) ) ) 
print("BLEU score on tags: {0:.0%}".format(round( score_bleu_tags), 4 ) ) 
print("BLEU score on activities: {0:.0%}".format(round( score_bleu_act), 4 ) )  
```



