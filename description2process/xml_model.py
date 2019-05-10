import tensorflow as tf
from tensor2tensor import problems
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import pickle
import requests, zipfile, io
import urllib
import os

## -- Set-up environment
#-----------------------

# We need to enable eager execution for inference at the end of this notebook.
"""
tfe = tf.contrib.eager
tfe.enable_eager_execution()

TFVERSION='1.13'
import os
os.environ['TFVERSION'] = TFVERSION
"""
# -- Download and unzip file to build model
def download_model():
    url = "https://www.dropbox.com/s/ucjup5di19xwelz/description2process.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

download_model()

# Include file sequence.py

import re
import string
import pandas as pd

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

@registry.register_problem
class sequence(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 2**13

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        dataset = pd.read_csv('/content/dummy_data_advanced.csv', sep=';')
        dataset.to_string()
        for index, row in dataset.iterrows():
            yield {
                "inputs": row['source'],
                "targets": row['target'],
                }

@registry.register_hparams
def transformer_tiny_test():
  hparams = transformer.transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.filter_size = 512
  hparams.num_heads = 4
  hparams.learning_rate=0.05
  #hparams.max_length=100
  hparams.batch_size=1000 #500
  return hparams

## -- Make model ready for predictions
#-------------------------------------

# specify arguments
problem_name = 'sequence'
model_name = 'transformer'
hparams_name = 'transformer_tiny_test'

data_dir = './description2process/data/'
tmp_dir = './description2process/data/tmp'
train_dir = './description2process/model/'  # this is the "output_dir" argument in the cli

# Fetch the problem
sequence_problem = problems.problem(problem_name)

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_name = "vocab.sequence.8192.subwords"
vocab_file = os.path.join(data_dir, vocab_name)
Modes = tf.estimator.ModeKeys

hparams = trainer_lib.create_hparams(hparams_name, data_dir=data_dir, problem_name=problem_name)
sequence_model = registry.model(model_name)(hparams, Modes.PREDICT)

## -- Internal functions
#-----------------------

# -- Tokenizer for transforming text to integers

# Create tokenizer with missing words
def get_tokenizer_with_missing_words(text, tk_defined):
    rm_chars = '!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n'

    missing_words = list()
    for word in text.lower().translate(str.maketrans('', '', rm_chars)).split():
      if not word in tk_defined.word_index.keys():
        missing_words.append(word)

    # Create tokenizer of missing words
    tk_missing = Tokenizer(filters = rm_chars)
    tk_missing.fit_on_texts(missing_words)

    # Increase all items of a dictionary with a given value
    def increase_key_value(value, word_index):
        word_index.update({key : word_index[key] + value for key in word_index.keys()})
        return word_index

    # Merge two dicitonaries together
    def merge_dictionaries(d1,d2):
        return {**d1, **d2}

    tk_missing.word_index = increase_key_value(1379, tk_missing.word_index)

    # Create tokenizer, which is combinaton of tk_defined and tk_missing
    tk = Tokenizer(filters = rm_chars)
    tk.word_index = merge_dictionaries(tk_defined.word_index, tk_missing.word_index)

    return tk

# INPUT: structured-description (text)
# OUTPUT: integer encoded structured description
def encode_text(text, tk):
  int_encoded = tk.texts_to_sequences([text])
  int_encoded = int_encoded[0]
  text_int = ""
  for i in int_encoded :
    text_int = text_int + " " + str(i)

  return text_int

# Decode list of integers to their original words.
# Dictionary of defined tokenizer is used for the decoding.
def decode_tokenizer(decode, tokenizer) :
    # reverse dictionary in tokenizer (index --> words)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    # Create list of strings
    decode = decode.split()
    # Convert strings into integers
    decode = list(map(int, decode))
    text = ""
    for integer in decode :
        if integer > 0 :
            text = text + " " + reverse_word_map[integer]

    return text

# -- Encoding and decoding of the transformers embedding

# Get the encoders from the problem
encoders = sequence_problem.feature_encoders(data_dir)

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.

    return {"inputs": batch_inputs}

def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]

    return encoders["inputs"].decode(np.squeeze(integers))


# -- Use transformer to translate integer-encoded semi-structured description into xml format
ckpt_path = tf.train.latest_checkpoint(train_dir)

def translate(inputs):
    encoded_inputs = encode(inputs)
    with tfe.restore_variables_on_create(ckpt_path):
      model_output = sequence_model.infer(encoded_inputs)["outputs"]
    return decode(model_output)

# -- Create estimator
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import decoding
from tensor2tensor.data_generators import text_encoder

estimator = trainer_lib.create_estimator(
      'transformer',
      hparams,
      trainer_lib.create_run_config(hparams),
      decode_hparams=None,
      use_tpu=False)

decode_hp = decoding.decode_hparams()

def decode_from_input(estimator,
                     inputs,
                     hparams,
                     decode_hp,
                     decode_to_file=None,
                     checkpoint_path=None):

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = sequence

  sorted_inputs = [inputs]

  num_sentences = 1  #len(sorted_inputs)
  num_decode_batches = 1  # (num_sentences - 1) // decode_hp.batch_size + 1

  def input_fn():
    input_gen = _decode_batch_input_fn(
        num_decode_batches,
        sorted_inputs,
        inputs_vocab,
        32, #decode_hp.batch_size,
        decode_hp.max_input_size)

    gen_fn = make_input_fn_from_generator(input_gen)
    example = gen_fn()

    return _decode_input_tensor_to_features_dict(example, hparams)

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  for num_predictions, prediction in enumerate(result_iter):
    inputs = prediction.get("inputs")
    targets = prediction.get("targets")
    outputs = prediction.get("outputs")

  # decode outputs
  out_int = decode(outputs)

  return out_int

# Own decode function

def decode(integers):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if 1 in integers:
      integers = integers[:integers.index(1)]

  return encoders["inputs"].decode(np.squeeze(integers))

# Function from tensor2tensor/decoding
import six

def make_input_fn_from_generator(gen):
  """Use py_func to yield elements from the given generator."""
  first_ex = six.next(gen)
  flattened = tf.contrib.framework.nest.flatten(first_ex)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [first_ex]

  def py_func():
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = six.next(gen)
    return tf.contrib.framework.nest.flatten(example)

  def input_fn():
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = tf.contrib.framework.nest.pack_sequence_as(first_ex, flat_example)
    return example

  return input_fn

def _decode_batch_input_fn(num_decode_batches, sorted_inputs, vocabulary,
                           batch_size, max_input_size,
                           task_id=-1, has_input=True):
  """Generator to produce batches of inputs."""
  #tf.logging.info(" batch %d" % num_decode_batches)
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      if has_input or task_id > -1:  # Do not append EOS for pure LM tasks.
        final_id = text_encoder.EOS_ID if task_id < 0 else task_id
        input_ids.append(final_id)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
    }

def _decode_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.
  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters
  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])
  input_is_image = False

  x = inputs
  p_hparams = hparams.problem_hparams
  # Add a third empty dimension
  x = tf.expand_dims(x, axis=[2])
  x = tf.to_int32(x)
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = (
      IMAGE_DECODE_LENGTH if input_is_image else tf.shape(x)[1] + 50)
  features["inputs"] = x
  return features
# -- Check if the prediction is in XML format
#--------------------------------------------

import xml.etree.ElementTree as ET
def xml_check(xml):
    try:
        tree_prediction = ET.fromstring("""<?xml version="1.0"?> <data> """ + prediction.strip() + " </data>")
        xml_checked = xml
    except:
        xml_checked = get_well_formed_xml(xml)

    return xml_checked

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


# -- Main functions
#------------------

# Load tokenizer
urllib.request.urlretrieve("https://www.dropbox.com/s/mk8p97hgdt0y9y6/tokenizer.pickle?dl=1", "tokenizer.pickle")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# -- Transform structured description into the xml format of a models
# INPUT: semi-structured-description
# OUTPUT: xml  format of model
def structured2xml_version2(text):
    tk = get_tokenizer_with_missing_words(text, tokenizer)
    # Integer encode description
    text_int = encode_text(text, tk).strip()
    # Apply transformer model
    out_int = translate(text_int)
    # Decode integer sequence to textual description
    decoded_text = decode_tokenizer(out_int, tk).strip()

    return decoded_text

def structured2xml(text):
    tk = get_tokenizer_with_missing_words(text, tokenizer)
    # Integer encode description
    text_int = encode_text(text, tk).strip()
    # Apply transformer model
    out_int = decode_from_input(estimator, text_int, hparams, decode_hp, checkpoint_path=ckpt_path)
    # Decode integer sequence to textual description
    decoded_text = decode_tokenizer(out_int, tk).strip()
    # Check if predition is in xml format
    xml_checked = xml_check(decoded_text)

    return decoded_text
