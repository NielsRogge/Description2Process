# Description2Process
This repository includes all data as well as source code of the paper "On the Applicability of Deep Learning to construct Process Models from Natural Text". 

The proposed methodology consists of an **NLP pipeline** that transforms a textual description into a process model in an XML format which eventually can be mapped to a graph that is roughly based on BPMN. The NLP pipeline consists of 8 sequential steps:

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
The project is based on Python 3. To install, simply do
```
pip install description2process
```

## Usage 

(...)
