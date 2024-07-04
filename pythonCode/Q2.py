from utilityFunctions import retrieve_document_vector_values as getDocVector
from ndcg import calculate_ndcg_for_ranking as calculate_ndcg_score
import pandas as pd
import os
import math
from math import log, log2
from collections import defaultdict


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
    idealTopK=idealRanking_sorted[:k]
    idealScores=[score for _,score in idealTopK]
    idealNormalizedScores=min_max_normalize(idealScores)
    idealTopK_ids=[doc[0]for doc in idealTopK]
    myRanking_filtered=[doc for doc in myRanking if doc[0]in idealTopK_ids]
    myRanking_sorted=sorted(myRanking_filtered,key=lambda x:x[1],reverse=True)
    myTopK=myRanking_sorted[:k]
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

relevance_folder = os.path.join('relevance')
relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')
relevance_data = load_relevance_data(relevance_file_path)
def calculate_ndcg_for_ranking(myRanking, query_id, k):
    idealRanking  = relevance_data[query_id]
    if(len(idealRanking)<k):
        print("Error: idealRanking does not have enough documents.")
        return
    idealTopK, myTopK = sort_and_select_top_k(myRanking, idealRanking, k)
    ndcg_score = ndcg_at_k(myTopK, idealTopK, k)
    if ndcg_score > 1:
        print("Error: Ranking does not have enough non-zero relevance documents.")
        # print(myTopK, idealTopK)
        return
    print("NDCG Score:", ndcg_score)


# taking choice from user
print("For document: enter 1 for nnn, 2 for ntn, 3 for ntc")
choice_doc = input("Enter your choice: ")

print("For query: enter 1 for nnn, 2 for ntn, 3 for ntc")
choice_query = input("Enter your choice: ")



# loading queries
queryList = []

def read_queries_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            queryList.append((query_id, query_text))

processed_queries_folder = os.path.join(os.getcwd(),'pythonCode', 'processedQueries')
files = ['combined_dev_queries.txt', 'combined_test_queries.txt', 'combined_training_queries.txt']

for file_name in files:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file(file_path)
    else:
        print(f"File not found: {file_path}")

# df = pd.DataFrame(queryList, columns=['Query ID', 'Query Text'])
# print(df.head())

# loading the list of docids
docid_list = []

def retrieve_docids_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid = line.strip().split('\t')[0]
            docid_list.append(docid)

final_data_file = os.path.join(os.getcwd(),'pythonCode', 'processedData', 'processedData.txt')

if os.path.exists(final_data_file):
    retrieve_docids_from_file(final_data_file)
else:
    print(f"File not found: {final_data_file}")


# def dot_product(vector1, vector2):
#     result = 0.0
#     for key in vector1:
#         result += vector1[key] * vector2.get(key, 0.0)
#     return result

# def get_word_value(vector, word):
#     return vector.get(word, 0.0)



# given a test, give its words
def extract_words_from_text(text):
    words = text.split()
    return words

# it gives df and corresponding docids
def load_index_combined(index_combined_file):
    index_combined = defaultdict(lambda: [0, set()])
    with open(index_combined_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, df, doc_ids = line.strip().split('\t')
            index_combined[word][0] = int(df)
            index_combined[word][1] = set(doc_ids.split())
    return index_combined

# it gives word and corresponding docids
def build_index_map(index_file):
    index_map = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word, _, doc_ids_str = parts[:3]
                doc_ids = set(doc_ids_str.split())
                if word not in index_map:
                    index_map[word] = set()
                index_map[word].update(doc_ids)
    return index_map

# searches whether a word exists in a doc
def check_word_in_document(word, doc_id, index_map):
    if word in index_map and doc_id in index_map[word]:
        return 1
    return 0

# get the doc value for a vector
def getdocValue(word):
    if choice_doc == '1':
        return 1.0
    else:
        df =  index_combined[word][0]
        return log(5371 / (df+1))
    
#  get the query value for a vector
def getqueryValue(word):
    if choice_query == '1':
        return 1.0
    else: 
        df = index_combined[word][0]
        return log(5371 / (df+1))

# does normalisation
def cosine_normalization_term(vector):
    squared_sum = sum(x ** 2 for x in vector)
    normalization_term = math.sqrt(squared_sum)
    return normalization_term

# def load_stopwords(stopwords_path):
#     with open(stopwords_path, 'r', encoding='utf-8') as file:
#         stopwords_list = file.read().splitlines()
#     return set(stopwords_list)

# current_directory = os.path.dirname(os.path.abspath(__file__))
# stopwords_path = os.path.join(current_directory, '..', 'stopWords', 'stopwords.large')
# stopwords_set = load_stopwords(stopwords_path)

def load_doc_word_mapping(file_path):
    doc_word_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc_id, words = line.strip().split('\t')
            doc_word_mapping[doc_id] = words.split()
    return doc_word_mapping

current_directory = os.path.dirname(os.path.abspath(__file__))
doc_word_file = os.path.join(current_directory, 'output', 'doc_word_list.txt')
doc_word_mapping = load_doc_word_mapping(doc_word_file)

# Load the index_combined and index_map
current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
index_combined = load_index_combined(index_combined_file)
index_map = build_index_map(index_combined_file)

# main task starts
for query in queryList:
    query_id, query_text = query
    ranking = {}
    for docid in docid_list:
        word_list = extract_words_from_text(query_text)
        similarity = 0.0
        doclist = []
        docwordlist = doc_word_mapping[docid]
        for word in docwordlist:
            docval = getdocValue(word)
            doclist.append(docval)
        querylist = []
        for queryWord in word_list:
            queryval = getqueryValue(queryWord)
            if(queryWord in index_map):
                querylist.append(queryval)
            value = check_word_in_document(queryWord, docid, index_map)
            if value != 0:
                docval = getdocValue(queryWord)
                # queryval = getqueryValue(queryWord)
                # doclist.append(docval)
                # querylist.append(queryval)
                similarity += value * docval * queryval
        if choice_doc == '3':
            doc_norm = cosine_normalization_term(doclist)
            if doc_norm != 0:
                similarity = similarity / doc_norm
        if choice_query == '3':
            query_norm = cosine_normalization_term(querylist)
            if query_norm!= 0:
                similarity = similarity / query_norm
        ranking[docid] = similarity

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    print(query_id)
    print(query_text)
    for doc_id, score in sorted_ranking[:10]:
         print(f"Document ID: {doc_id}, Score: {score}")
    k = 10
    calculate_ndcg_for_ranking(sorted_ranking, query_id, k)
    # break