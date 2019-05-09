import spacy
import neuralcoref
spacy.cli.download("en")

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

#import en_coref_lg
#nlp = en_coref_lg.load()

# List of references that will be resolved
references = ['he','she','it']

# Returns the coreference resolved descriptions
# Input: pandas dataframe with descriptions to resolve
# Output: pandas dataframe with the descriptions replaced with the resolved version
def resolve_coreferences_df(descriptions):
    descriptions['description'] = descriptions.apply(lambda row: get_clean_description(row['description']),axis=1)
    return descriptions

# Resolve the coreferences of one text (description)
# Input: string
def resolve_coreferences(text) :
    return get_clean_description(text)

# Returns the resolved sentences in string format
# Input: spacy document with the sentences that need to be resolved
def resolve_doc(doc):
  if doc._.has_coref == False:
    resolved_sentence = str(doc)
  else:
    list_coreferences = []
    for cluster in doc._.coref_clusters:
      list_coreferences.append(str(cluster.mentions[-1]).lower())

    if any(elem in list_coreferences for elem in references) :
      resolved_sentence = doc._.coref_resolved
    else:
      resolved_sentence = str(doc)
  return resolved_sentence

def resolve_text(text):
    doc = nlp(text)

    if doc._.has_coref == False:
      resolved_sentence = str(doc)
    else:
      list_coreferences = []
      for cluster in doc._.coref_clusters:
        list_coreferences.append(str(cluster.mentions[-1]).lower())

      if any(elem in list_coreferences for elem in references) :
        resolved_sentence = doc._.coref_resolved
      else:
        resolved_sentence = str(doc)
    return doc, resolved_sentence


def get_clean_description(text):

  # Initialize clean text
  clean_text = ""

  # Create doc object from the complete descriptions
  doc = nlp(text)

  # Create list of all sentences
  sents = list(doc.sents)

  # Resolve first sentence and add it to the clean text
  doc1 = nlp(str(sents[0]))
  resolved_sentence1 = resolve_doc(doc1)

  sents[0] = resolved_sentence1
  clean_text = clean_text + " " + resolved_sentence1

  #use the resolved previous sentence to resolve the next sentence

  for j in range(1, len(sents)):
    # get sentence 1
    sentence_prev = str(sents[j-1])
    # get sentence 2
    sentence = str(sents[j])

    # create doc of both sentence 1 and 2 togehter
    #doc_j = nlp(sentence_i + " " + sentence_j)

    # returns string and doc of both sentences resolved
    #resolved_sentence_j = resolve_doc(doc_j)

    resolved_doc, resolved_text = resolve_text(sentence_prev + " " + sentence)
    resolved_doc = nlp(resolved_text)

    # add resolved sentence to the list of all sentences and
    # also add the resolved sentence to the clean text
    sents[j] = str(list(resolved_doc.sents)[-1])
    clean_text = clean_text + " " + sents[j]

  return clean_text.strip()
