import os
import numpy as np
import math
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import re

# Read test queries and documents
def read_queries(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip().split('\t') for line in file]

def read_documents(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip().split('\t') for line in file]

# Preprocess text
def preprocess(text):
    tokens = re.findall(r'\w+', text.lower()) # Tokenize and lowercase
    # Other preprocessing steps like removing stop words, stemming, etc.
    return tokens


def generate_tf_df(documents):
    tf_dict = defaultdict(dict)  # {doc_id: {word: tf}}
    df_dict = defaultdict(int)   # {word: df}

    for doc_id, doc_content in documents:
        doc_tf = {}  # Initialize a dictionary to store TF for the current document

        # Calculate TF for each word in the document
        for word in doc_content:
            doc_tf[word] = doc_tf.get(word, 0) + 1

        # Update TF dictionary with TF values for the current document
        tf_dict[doc_id] = doc_tf

        # Update DF dictionary with DF values for each word in the current document
        for word in set(doc_content):
            df_dict[word] += 1

    return tf_dict, df_dict

def calculate_rsv(query, doc_id, tf_dict, df_dict, N, avg_doc_length, k1=1.2, b=0.75):
    rsv = 0
    
    doc_tf = tf_dict[doc_id]
    doc_length = sum(doc_tf.values())

    # # Assuming doc_id is the document ID for which you want to print doc_tf
    # doc_tf = tf_dict[doc_id]
    # print("Document ID:", doc_id)
    # print("Term Frequencies:")
    # for term, frequency in doc_tf.items():
    #     print(f"{term}: {frequency}")


    for word in query:
        f_qi_d = doc_tf.get(word, 0)  # Get the actual term frequency
        df = df_dict.get(word, 0)
        idf = math.log((N + 0.5) / (df + 0.5))  # Corrected IDF formula
        term1 = ((k1 + 1) * f_qi_d) / (f_qi_d + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        # print(f_qi_d, idf, term1)
        term2 = idf
        rsv += term1*term2
    return rsv

# Retrieve top k documents for each query
def retrieve_top_documents(queries, tf_dict, df_dict, avg_doc_length, N, k=5):
    top_documents = defaultdict(list)
    
    for query_id, query in queries:
        for doc_id, doc_tf in tf_dict.items():
            rsv = calculate_rsv(query, doc_id, tf_dict, df_dict, N, avg_doc_length)
            top_documents[query_id].append((doc_id, rsv))
        
        top_documents[query_id].sort(key=lambda x: x[1], reverse=True)
        top_documents[query_id] = top_documents[query_id]
    
    return top_documents


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
        # print("Error: idealRanking does not have enough documents.")
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
        # print("Error: idealRanking does not have enough documents.")
        return -1
    idealTopK, myTopK = sort_and_select_top_k(myRanking, idealRanking, k)
    ndcg_score = ndcg_at_k(myTopK, idealTopK, k)
    if ndcg_score > 1:
        # print("Error: Ranking does not have enough non-zero relevance documents.")
        return -1
    # print("NDCG Score:", ndcg_score)
    return ndcg_score

# Write results to file
def write_results_to_file(queries, top_documents, output_file,k):
    with open(output_file, "w") as f:
        for query_id, query in queries:
            ndcg_score =  calculate_ndcg_for_ranking(top_documents[query_id], query_id, 5)
            f.write(f"Query: {query_id} - {query} - {ndcg_score}\n")
            for rank, (doc_id, rsv) in enumerate(top_documents[query_id][:k], 1):
                f.write(f"Rank {rank}: Document {doc_id} - RSV: {rsv}\n")
        f.write("\n")

if __name__ == "__main__":
    # Read data
    queries = read_queries('pythonCode/processedQueries/test_queries.txt')
    doc_ids = read_documents('pythonCode/output/doc_word_list.txt')
    N = len(doc_ids)

    # Preprocess queries
    preprocessed_queries = [(query_id, preprocess(query)) for query_id, query in queries]

    # Preprocess documents
    documents = [(doc_id, preprocess(doc)) for doc_id, doc in doc_ids]

    # Calculate TF and DF values
    tf_dict, df_dict = generate_tf_df(documents)

    # Calculate average document length
    avg_doc_length = np.mean([len(doc) for _, doc in documents])
    print("Average document length:", avg_doc_length)

    # Retrieve top documents for each query
    top_documents = retrieve_top_documents(preprocessed_queries, tf_dict, df_dict, avg_doc_length, N)

    # Write results to file
    output_file = "pythonCode/output/output_bm25.txt"
    k=5
    write_results_to_file(preprocessed_queries, top_documents, output_file,k)



    # # Print tf_dict
    # for doc_id, doc_tf in tf_dict.items():
    #     print(f"Document ID: {doc_id}")
    #     for word, tf in doc_tf.items():
    #         print(f"  Word: {word}, TF: {tf}")

    # # Print df_dict
    # print("DF Dictionary:") # Debugging: Print DF dictionary
    # for word, df in df_dict.items():
    #     print(f"  Word: {word}, DF: {df}")

    # def write_dicts_to_file(tf_dict, df_dict, output_file):
    #     with open(output_file, "w") as f:
    #         f.write("tf_dict:\n")
    #         for doc_id, doc_tf in tf_dict.items():
    #             f.write(f"Document {doc_id}: {doc_tf}\n")
            
    #         f.write("\ndf_dict:\n")
    #         for word, df in df_dict.items():
    #             f.write(f"{word}: {df}\n")

    #     # Example usage:
    #     output_file = "pythonCode/output/output_dicts.txt"
    #     write_dicts_to_file(tf_dict, df_dict, output_file)