import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = pd.read_json('data/documents.json')
queries = pd.read_json('data/queries.json')
qrels = pd.read_json('data/qrels.json')

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents['text'])

print("Calculating TF-IDF matrix:\n")
print(pd.DataFrame(data=tfidf_matrix.toarray(), index=documents['doc_id'], columns=vectorizer.get_feature_names_out()))
print("Finished calculating TF-IDF matrix.\n")

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

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
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
    
    # store the AP value for this query (use any data structure you prefer)
    total_ap_score = precision_sum / relevant_doc_count if relevant_doc_count > 0 else 0
    ap_scores[query['query_id']] = total_ap_score
        
    print("Relevant doc count: {}".format(relevant_doc_count))
    print("Irrelevant doc count: {}".format(irrelevant_doc_count))
    print("Total AP for {}: {}".format(query['query_id'], total_ap_score))
    print()

print("Static Query MAP Score: {}".format(sum(ap_scores.values()) / len(ap_scores)))