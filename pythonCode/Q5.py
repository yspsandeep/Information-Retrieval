import re
import csv
from collections import defaultdict
import numpy as np
import os

# Load the Knowledge Graph (GENA) from the CSV file
knowledge_graph = defaultdict(list)
with open('gena_data_final_triples.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        subject, relation, obj = row
        knowledge_graph[subject].append((relation, obj))
        knowledge_graph[obj].append((relation, subject))

queryList = []

def read_queries_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            queryList.append((query_id, query_text))

files = ['/content/combined_dev_queries.txt', '/content/combined_test_queries.txt', '/content/combined_training_queries.txt']

for file_name in files:
    read_queries_from_file(file_name)

# Loading documents
documents = {}

def load_documents_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid, text = line.strip().split('\t', 1)
            documents[docid] = text

load_documents_from_file('/content/processedData.txt')

# Load stop words from the file
with open('/content/stopwords.large', 'r') as f:
    stop_words = set(line.strip() for line in f)


import spacy
from collections import Counter

# Load the pre-trained SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    # Create a SpaCy Doc object
    doc = nlp(text)
    
    # Extract all nouns as entities
    entities = [token.text for token in doc if token.pos_ == "NOUN"]
    
    return entities

def bag_of_entities(text):
    entities = extract_entities(text)
    return Counter(entities)

def coordinate_match(query_entities, doc_entities):
    common_entities = query_entities & doc_entities
    query_score = sum(query_entities.values())
    doc_score = sum(doc_entities.values())
    
    if query_score == 0 and doc_score == 0:
        return 0  # Return 0 if both query and document entities are empty
    else:
        return sum(common_entities.values()) / (query_score + doc_score - sum(common_entities.values()))

def entity_frequency_score(query_entities, doc_entities):
    return sum((query_entities & doc_entities).values()) / max(sum(query_entities.values()), sum(doc_entities.values()))

def compute_similarity(query_entities, doc_entities):
    coord_match_score = coordinate_match(query_entities, doc_entities)
    entity_freq_score = entity_frequency_score(query_entities, doc_entities)
    return (coord_match_score + entity_freq_score) / 2


import math

def load_relevance_data(file_path):
    relevance_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                relevance_data[query_id] = []
            relevance_data[query_id].append((doc_id, relevance_score))
    return relevance_data

def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)  # Set all scores to 1 if min and max are the same
    normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores


def sort_and_select_top_k(myRanking,idealRanking,k):
    idealRanking_sorted=sorted(idealRanking,key=lambda x:x[1],reverse=True)
    if len(idealRanking_sorted)<k:
        print("Error: idealRanking does not have enough documents.")
        return None,None
    idealTopK=idealRanking_sorted
    idealScores=[score for _,score in idealTopK]
    idealNormalizedScores=min_max_normalize(idealScores)
    idealTopK_ids=[doc[0]for doc in idealTopK]
    myRanking_filtered=[doc for doc in myRanking if doc[0]in idealTopK_ids]
    myRanking_sorted=sorted(myRanking_filtered,key=lambda x:x[1],reverse=True)
    myTopK=myRanking_sorted
    myScores=[score for _,score in myTopK]
    myNormalizedScores=min_max_normalize(myScores)
    return list(zip(idealTopK_ids,idealNormalizedScores)),list(zip([doc[0]for doc in myTopK],myNormalizedScores))

def dcg_at_k(ranking, k):
    dcg = ranking[0][1]
    for i in range(1, min(k, len(ranking))):
        dcg += ranking[i][1] / math.log2(i + 1)
    return dcg

def ndcg_at_k(ranking, ideal_ranking, k):
    ideal_dcg = dcg_at_k(ideal_ranking, k)
    if ideal_dcg == 0:
        return 0
    return dcg_at_k(ranking, k) / ideal_dcg

def calculate_ndcg_for_ranking(myRanking, query_id, k):
    #relevance_folder = os.path.join('relevance')
    relevance_file_path = '/content/merged.qrel'
    relevance_data = load_relevance_data(relevance_file_path)
    idealRanking  = relevance_data[query_id]
    if(len(idealRanking)<k):
        print("Error: idealRanking does not have enough documents.")
        return
    idealTopK, myTopK = sort_and_select_top_k(myRanking, idealRanking, k)
    ndcg_score = ndcg_at_k(myTopK, idealTopK, k)
    if ndcg_score > 1:
        print("Error: Ranking does not have enough non-zero relevance documents.")
        return
    print("NDCG Score:", ndcg_score)

def retrieve_documents(query, documents):
    query_entities = bag_of_entities(query)
    scores = []

    for doc_id, doc_text in documents.items():
        doc_entities = bag_of_entities(doc_text)
        similarity_score = compute_similarity(query_entities, doc_entities)
        scores.append((doc_id, similarity_score))

    # Sort the documents by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def retrieve_and_print_documents(queryList, documents, num_queries=10):
    for i, query in enumerate(queryList[:num_queries]):
        query_text = query[1] if isinstance(query, tuple) else query
        query_entities = extract_entities(query_text)

        print(f"Query: {query_text}")
        print(f"Query Entities: {query_entities}")

        scores = retrieve_documents(query_text, documents)

        query_id = query[0]
        calculate_ndcg_for_ranking(scores,query_id,5)

        if not scores:
            print("No relevant documents found.")
        else:
            print("Top 5 relevant documents:")
            for doc_id, similarity_score in scores[:5]:
                doc_text = documents[doc_id]
                print(f"Document ID: {doc_id}, Score: {similarity_score:.4f}")
                print(f"Document Text: {doc_text}")
                print()

        if i < num_queries - 1:
            print("-" * 80)

retrieve_and_print_documents(queryList, documents, num_queries=10)
