import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = pd.read_json('data/documents.json')
queries = pd.read_json('data/queries.json')
qrels = pd.read_json('data/qrels.json')

vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform(documents['text'])

print("Calculating TF-IDF matrix:\n")
print(pd.DataFrame(data=tfidf_matrix.toarray(), index=documents['doc_id'], columns=vectorizer.get_feature_names_out()))
print("Finished calculating TF-IDF matrix.\n")

print("Writing inverted index to 'inverted_index.json'\n")
with open('inverted_index.json', 'w') as f:
  inverted_index = {}

  for term in vectorizer.get_feature_names_out():
    term_index = vectorizer.vocabulary_[term]
    doc_indices = tfidf_matrix[:, term_index].nonzero()[0]

    doc_ids = list(set(documents['doc_id'][doc_indices]))
    tfidf_vals = tfidf_matrix[doc_indices, term_index].toarray().flatten()
    doc_ids_tfidf_zipped = list(zip(doc_ids, tfidf_vals))

    inverted_index[term] = doc_ids_tfidf_zipped
  
  json.dump(inverted_index, f, indent=2)
print("Finished writing inverted index.\n")

print("Writing keyed documents file to 'keyed_documents.json'\n")
with open('keyed_documents.json', 'w') as f:
  keyed_documents = {}
  for idx, doc in documents.iterrows():
    keyed_documents[doc['doc_id']] = doc['text']
  json.dump(keyed_documents, f, indent=2)
print("Finished writing keyed documents file.\n")

ap_scores = {}

for idx, query in queries.iterrows():
  query_vector = vectorizer.transform([query['text']])
  similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
  zip_similarities = list(zip(documents['doc_id'], similarities))
  zip_similarities.sort(key=lambda x: x[1], reverse=True)
  sorted_doc_ids = [doc_id for doc_id, _ in zip_similarities]
  top25 = sorted_doc_ids[:25]

  print("Sorted Cosine Similarities for {}:".format(query['query_id']))
  print(top25)
  print()

  relevance_entries = qrels[qrels['query_id'] == query['query_id']]
  relevant_doc_ids = set(relevance_entries['doc_id'])
  print("Total relevant entries for query {}: {}".format(query['query_id'], len(relevant_doc_ids)))
  relevant_doc_count = 0
  irrelevant_doc_count = 0
  precision_sum = 0

  for doc_id in top25:
    if doc_id in relevant_doc_ids:
      relevant_doc_count += 1
      precision_sum += relevant_doc_count / (relevant_doc_count + irrelevant_doc_count)
    else:
      irrelevant_doc_count += 1
  
  total_ap_score = precision_sum / (relevant_doc_count if relevant_doc_count > 0 else 1)
  ap_scores[query['query_id']] = total_ap_score
      
  print("Relevant doc count: {}".format(relevant_doc_count))
  print("Irrelevant doc count: {}".format(irrelevant_doc_count))
  print("Total AP for {} ({}): {}".format(query['query_id'], query['text'], total_ap_score))
  print()

print("Static Query MAP Score: {}".format(sum(ap_scores.values()) / len(ap_scores)))