# Description2Process
This repository includes all data as well as source code of the paper "On the Applicability of Deep Learning to construct Process Models from Natural Text". 

The proposed methodology consists of an **NLP pipeline** that transforms a textual description into a process model in an XML format, which eventually can be mapped to a graph that is roughly based on BPMN. The NLP pipeline consists of 8 sequential steps:

1. Contraction expansion
2. Coreference resolution
3. Clause extraction
4. Activity recognition
5. Activity extraction
6. Construction of semi-structured description
7. Transformation of semi-structured description to XML format
8. Visualization of XML format 

Textual inputs are limited to sequences of activities and 3 types of XOR splits.

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
To start, first write a textual description yourself: 
```
description = "First, the customer purchases a product. Then, the customer receives an invoice. Finally, the customer pays the bill."
```
Alternatively, one can let our data generation algorithm create a random one based on the 120 scenarios mentioned in the paper:
```
from description2process import data_generation
description_df = d2p.data_generation.get_descriptions(ndescriptions = 3, nscenario = 70, start = 0, data_frame = True)
description_df.tail()
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
To split up a given description into separate clauses, the clause extraction algorithm can be called as follows:
```
clauses = d2p.clause_extraction.get_clauses(description)
print(clauses)
```

### Activity recognition
Next, activity recognition can be performed on the extracted clauses. This involves a deep neural network classifying each clause separately. The result is a list of labeled clauses, where "True" indicates that the clause includes an activity, whereas "False" indicates that the clause does not include an activity. Activity recognition can be done as follows:
```
labeled_clauses = d2p.activity_recognition.contains_activity_list(clauses)
print(labeled_clauses)
```

### Activity extraction
To extract the activities of the clauses that contain an activity, the activity extraction algorithm can be called as follows:
```
extracted_activities = d2p.activity_extraction.get_activity_df(labeled_clauses)
print(extracted_activities)
```
This returns a Pandas dataframe, including 3 columns: the extracted clauses, labels indicating whether or not each clause contains an activity or not, and the extracted activities.  

### Construction of semi-structured description
Constructing a semi-structured description (based on the description cleaned during the contraction expansion and coreference resolution steps together with the extracted activities) can be done as follows:
```
structured_description = d2p.structured_description.get_structured_description(description, extracted_activities)
print(structured_description)
```

### Transformation of semi-structured description to XML format
Next, a Transformer model can be called to translate the semi-structured description to an XML format that represents the process model. This is done as follows:
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



