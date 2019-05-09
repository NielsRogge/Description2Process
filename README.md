# Description2Process
This repository includes all data as well as source code of the paper "On the Applicability of Deep Learning to construct Process Models from Natural Text". 

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
The project requires Python 3. To install, simply do
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
description = "First the secretary clarifies the shipment method. Whenever there's a requirement for large amounts, he selects one of three logistic companies. Whenever small amounts are required, a package label is written by the secretary. Subsequently the goods can be packaged by the warehousemen. If everything is ready, the packaged goods are prepared for being picked up by the logistic company."
```
Alternatively, one can let our data generation algorithm create a random description based on the 120 scenarios mentioned in the paper:
```
from description2process import data_generation
description = d2p.data_generation.get_description()
print(description)
```

### Contraction expansion 
Next, one can expand all contractions of the description, like "doesn't" and "isn't" to "does not" and "is not". This happens as follows:
```
description = d2p.contraction_expansion.expand_contractions(description)
print(description)
```

### Coreference resolution 
Subsequently, coreference resolution can be performed, resolving references like "he", "she" and "it". This can be done as follows:
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
labeled_clauses = d2p.activity_recognition.contains_activity_list(clauses)
print(labeled_clauses)
```

### Activity extraction
To extract the activities of the clauses that contain an activity, the activity extraction algorithm can be called. This simply adds a column to the Pandas dataframe obtained in the previous step that includes the extracted activities. For clauses that do not contain an activity, "NaN" is displayed. The activity extraction algorithm can be called as follows:
```
extracted_activities = d2p.activity_extraction.get_activity_df(labeled_clauses)
print(extracted_activities)
```  

### Construction of semi-structured description
Next, one can construct a semi-structured description, which replaces the clauses that contained an activity from the original description (cleaned during the contraction expansion and coreference resolution steps) by their extracted activities, surrounded by <act> and </act> tags. This can be done as follows:
```
structured_description = d2p.structured_description.get_structured_description(description, extracted_activities)
print(structured_description)
```

### Transformation of semi-structured description to XML format
Next, a Transformer model can be called to translate the semi-structured description to a complete XML format that represents the process model. If appropriate, the model should add <path> and </path> tags to passages describing splits and merges. The Transformer model can be called as follows:
```
xml = d2p.xml_model.structured2xml(structured_description)
print(xml)
```

### Visualization of generated XML format 
Finally, the generated XML format can be mapped to an image (graph) as follows:
```
image = d2p.visualization.xml2model(xml)
print(image)
```



