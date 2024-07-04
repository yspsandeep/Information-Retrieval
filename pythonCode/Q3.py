from collections import defaultdict
from math import log
import math
import os
from utilityFunctions import retrieve_document_vector_values as getDocVector


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
# files = ['test_queries.txt']

for file_name in files:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file(file_path)
    else:
        print(f"File not found: {file_path}")



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


# retrieve relevence data given a queryid
def load_relevance_data(file_path):
    relevance_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                relevance_data[query_id] = []
            relevance_data[query_id].append((doc_id, relevance_score))
    return relevance_data

relevance_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'relevance')
file_path = os.path.join(relevance_folder, 'merged.qrel')
relevance_data = load_relevance_data(file_path)

def get_documents_with_scores(query_id):
    if query_id in relevance_data:
        return relevance_data[query_id]
    else:
        return []

def extract_words_from_text(text):
    words = text.split()
    return words

current_directory = os.path.dirname(os.path.abspath(__file__))
doc_word_file = os.path.join(current_directory, 'output', 'doc_word_list.txt')

def load_doc_word_mapping(file_path):
    doc_word_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc_id, words = line.strip().split('\t')
            doc_word_mapping[doc_id] = words.split()
    return doc_word_mapping

doc_word_mapping = load_doc_word_mapping(doc_word_file)

def get_words(doc_id):
    return doc_word_mapping.get(doc_id, [])

# sum of vectors
def sum_document_vectors(relevant_docs):
    summed_vector = create_zero_vector(index_map)
    for doc_id in relevant_docs:
        words = get_words(doc_id)
        for word in words:
            summed_vector[word] += 1
    return sorted(summed_vector.items())

def compute_difference(relevant_vector, non_relevant_vector, beta, gamma):
    difference_vector = create_zero_vector(index_map)
    
    # Iterate over relevant vector
    for word, value in relevant_vector:
        difference_vector[word] = beta * value
    
    # Subtract non-relevant vector
    for word, value in non_relevant_vector:
        difference_vector[word] -= gamma * value
    
    return sorted(difference_vector.items())

def add_alpha_to_vector(vector, query_words, alpha):
    updated_vector = vector.copy()  # Create a copy of the original vector
    
    if isinstance(updated_vector, list):
        # If the vector is a list, convert it to a dictionary
        updated_vector = {item[0]: item[1] for item in updated_vector}
    
    for word in query_words:
        if word in updated_vector:
            updated_vector[word] += alpha
    
    # Make any negative values zero in the updated vector
    updated_vector = {word: max(value, 0) for word, value in updated_vector.items()}
    
    return updated_vector




def non_zero_words(vector):
    non_zero_dict = {}
    for word, value in vector.items():
        if value != 0:
            non_zero_dict[word] = value
    return non_zero_dict

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

# Load the index_combined and index_map
current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
index_combined = load_index_combined(index_combined_file)
index_map = build_index_map(index_combined_file)

# searches whether a word exists in a doc
def check_word_in_document(word, doc_id, index_map):
    if word in index_map and doc_id in index_map[word]:
        return 1
    return 0

def create_zero_vector(index_map):
    sorted_words = sorted(index_map.keys())  # Sort the words
    zero_vector = {word: 0 for word in sorted_words}
    return zero_vector

# get the doc value for a vector
def getdocValue(word):
    if choice_doc == '1':
        return 1.0
    else:
        df =  index_combined[word][0]
        return log(5371 / (df + 1))
    
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

# main task starts here:
for query in queryList:
    query_id, query_text = query
    queryWords = extract_words_from_text(query_text)
    ranking = {}
    for docid in docid_list:
        # word_list = extract_words_from_text(query_text)
        similarity = 0.0
        # doclist = []
        # querylist = []
        for queryWord in queryWords:
            value = check_word_in_document(queryWord, docid, index_map)
            if value != 0:
                docval = 1
                queryval = 1
                # doclist.append(docval)
                # querylist.append(queryval)
                similarity += value * docval * queryval
        ranking[docid] = similarity

    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    k = 20
    topK_sorted_ranking = sorted_ranking[:k]
    documents = [item[0] for item in topK_sorted_ranking]
    relevant_docs = documents[:k]
    non_relevant_docs = []
    # relevant_docs = documents[:20]
    # non_relevant_docs = documents[20:]
    lenRel = len(relevant_docs)
    lenNonRel = len(non_relevant_docs)
    alpha = 1
    beta = 0.75
    gamma = 0.25
    if(lenRel!=0):
        beta = beta/lenRel
    if(lenNonRel!=0):
        gamma = gamma/lenNonRel
    relevant_vectors = sum_document_vectors(relevant_docs)
    nonrelevant_vectors = sum_document_vectors(non_relevant_docs)
    difference = compute_difference(relevant_vectors, nonrelevant_vectors, beta, gamma)
    updatedQuery = add_alpha_to_vector(difference, queryWords, alpha)
    non_zero_words_dict = non_zero_words(updatedQuery)
    # print(non_zero_words_dict)
    final_ranking = {}
    for docid in docid_list:
        new_similarity = 0
        doclist = []
        docwordlist = doc_word_mapping[docid]
        for word in docwordlist:
            docval = getdocValue(word)
            doclist.append(docval)
        querylist = []
        for word, value in non_zero_words_dict.items():
            queryval = getqueryValue(word)
            if(word in index_map):
                querylist.append(queryval)
            value = check_word_in_document(word, docid, index_map)
            if value != 0:
                docval = getdocValue(word)
                # queryval = getqueryValue(word)
                # doclist.append(docval)
                # querylist.append(queryval)
                new_similarity += docval * queryval * non_zero_words_dict[word]
        if choice_doc == '3':
            doc_norm = cosine_normalization_term(doclist)
            if doc_norm != 0:
                new_similarity = new_similarity / doc_norm
            # print("doing")
        if choice_query == '3':
            query_norm = cosine_normalization_term(querylist)
            if query_norm!=0:
                new_similarity = new_similarity / query_norm 
        final_ranking[docid] = new_similarity
    # final_sorted_ranking = sorted(final_ranking.items(), key=lambda x: (-x[1], x[0]))  
    final_sorted_ranking = sorted(final_ranking.items(), key=lambda x: x[1], reverse=True)  

    print(query_id)
    print(query_text)
    # for doc_id, score in sorted_ranking[:10]:
    #     print(f"Document ID: {doc_id}, Score: {score}")
    # print()
    for doc_id, score in final_sorted_ranking[:10]:
        print(f"Document ID: {doc_id}, Score: {score}")
    k = 10
    calculate_ndcg_for_ranking(final_sorted_ranking, query_id, k)
    # break