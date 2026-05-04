import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

while True:
  query = input("Enter your search query:\n")

  vectorizer = TfidfVectorizer(stop_words='english')
  query_vector = vectorizer.fit_transform([query])

  print("Query Vector:")
  print(pd.DataFrame(data=query_vector.toarray(), columns=vectorizer.get_feature_names_out()))
  # print(query_vector)

  top25 = list(range(25))
  selected_start = 1
  selected_end = 5

  while True:
    print("{} Results ({} - {})".format(query, selected_start, selected_end))
    for i in range(selected_start - 1, selected_end):
      print("[{}]: Document {}".format(i + 1, top25[i]))
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
      if selected_end < len(top25):
        selected_start += 5
        selected_end += 5
    
    print()

