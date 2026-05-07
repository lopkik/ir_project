import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

USE_INVERTED_INDEX = True

documents = pd.read_json('data/documents.json')
keyed_documents = pd.read_json('keyed_documents.json', typ='series')
inverted_index = pd.read_json('inverted_index.json', typ='series')

vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)

def highlight_query_terms(text, query_terms):
  highlighted_text = text
  for term in query_terms:
    highlighted_text = re.sub(r'\b{}\b'.format(re.escape(term)), "\033[92m{}\033[0m".format(term), highlighted_text, flags=re.IGNORECASE)
  return highlighted_text

while True:
  query = input("Enter your search query:\n")
  query_terms = []

  sorted_doc_ids = []

  if USE_INVERTED_INDEX:
    query_vector = vectorizer.fit_transform([query])
    query_terms = vectorizer.get_feature_names_out()
    print("Query Term TF-IDF Values: {}".format(query_terms))

    doc_vectors = {}
    for i, term in enumerate(query_terms):
      if term in inverted_index:
        for doc_id, tfidf_val in inverted_index[term]:
          if doc_id not in doc_vectors:
            doc_vectors[doc_id] = [0] * len(query_terms)
          doc_vectors[doc_id][i] = tfidf_val
    # print("Document Vectors for Query Terms:", doc_vectors)

    doc_similarities = []
    for doc_id, vector in doc_vectors.items():
      similarity = cosine_similarity([vector], query_vector.toarray()).flatten()[0]
      doc_similarities.append((doc_id, similarity.item()))
    doc_similarities.sort(key=lambda x: x[1], reverse=True)
    sorted_doc_ids = [(doc_id, similarity) for doc_id, similarity in doc_similarities]
  else:
    tfidf_matrix = vectorizer.fit_transform(documents['text'])
    query_vector = vectorizer.transform([query])
    query_term_indices = [vectorizer.vocabulary_.get(term) for term in query.split() if term in vectorizer.vocabulary_]
    query_terms = vectorizer.get_feature_names_out()[query_term_indices]

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    zip_similarities = list(zip(documents['doc_id'], similarities))
    zip_similarities.sort(key=lambda x: x[1], reverse=True)
    sorted_doc_ids = [(doc_id, similarity) for doc_id, similarity in zip_similarities]
    
  top25 = sorted_doc_ids[:25]

  selected_start = 1
  selected_end = 5

  while True:
    print("'{}' Query Results ({} - {})".format(query, selected_start, selected_end))
    for i in range(selected_start - 1, selected_end):
      if i >= len(top25):
        break
      print("\033[92m[{}]: Document {}, Score {}".format(i + 1, top25[i][0], top25[i][1]))
      highlighted_text = highlight_query_terms(keyed_documents[top25[i][0]], query_terms)
      print("\033[0m|-> {}".format(highlighted_text))
    print()
    print("'s' -> search with a new query; 'q' -> quit")
    print("'d' -> next page; 'a' -> previous page")
    
    print('----------------------------------------------------------------\n')
    command = input("Enter command: ")
    if command == 's':
      break
    elif command == 'q':
      exit()
    elif command == 'a':
      if selected_start > 1:
        selected_start -= 5
        selected_end -= 5
    elif command == 'd':
      if selected_end >= len(top25):
        print("No more results to display.\n")
        continue
      if selected_end < len(top25):
        selected_start += 5
        selected_end += 5
    
    print()

