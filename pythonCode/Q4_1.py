import os
import numpy as np
import math
from collections import Counter
from collections import defaultdict
from nltk.tokenize import word_tokenize

# Read test queries and documents
def read_queries(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip().split('\t') for line in file]

def read_documents(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [line.strip().split('\t') for line in file]

# Preprocess text
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    # Other preprocessing steps like removing stop words, stemming, etc.
    return tokens

# generate document word freq and length
def generate_document_word_frequency_and_length(documents):
    doc_word_freq = defaultdict(dict)
    doc_lengths = {}
    
    for doc_id, content in documents:
        tokens = preprocess(content)
        word_freq = defaultdict(int)
        
        for token in tokens:
            word_freq[token] += 1
        
        doc_word_freq[doc_id] = word_freq
        doc_lengths[doc_id] = len(tokens)
    
    return doc_word_freq, doc_lengths

def calculate_word_probability(term_frequency, doc_length, total_corpus_frequency, total_corpus_length, smoothing_lambda):
    # Probability of finding the word in the document using mixture model
    prob_doc = smoothing_lambda * (term_frequency / doc_length) + (1 - smoothing_lambda) * (total_corpus_frequency / total_corpus_length)
    return prob_doc

# def calculate_query_probability(query, doc_word_freq, doc_lengths, total_corpus_frequencies, total_corpus_length, smoothing_lambda):
#     # Initialize query log probability
#     query_log_prob = 0.0

#     # Iterate through each word in the query
#     for word in query.split():
#         term_frequency = doc_word_freq.get(word, 0)
#         word_prob = calculate_word_probability(term_frequency, total_corpus_length, total_corpus_frequencies.get(word, 0), total_corpus_length, smoothing_lambda)
        
#         # Add logarithm of word probability to query log probability
#         if(word_prob == 0):
#             word_prob = 1  # to avoid log(0)
#         query_log_prob += math.log(word_prob)

#     return query_log_prob

def calculate_query_probability(query, doc_term_frequencies, doc_length, total_corpus_frequencies, total_corpus_length, smoothing_lambda):
    # Initialize query probability
    query_prob = 1.0

    # Iterate through each word in the query
    for word in query.split():
        if word in doc_term_frequencies:
            term_frequency = doc_term_frequencies[word]
        else:
            term_frequency = 0
        # Calculate word probability
        word_prob = calculate_word_probability(term_frequency, doc_length, total_corpus_frequencies.get(word, 0), total_corpus_length, smoothing_lambda)
        # Multiply word probability to query probability
        if word_prob == 0:
            word_prob = 1  # to avoid log(0)
        query_prob *= word_prob

    return query_prob


def retrieve_documents(query, documents, doc_word_freq, doc_lengths, total_corpus_frequencies, total_corpus_length, smoothing_lambda, k=5):
    scores = []
    for doc_id, content in documents:
        term_frequencies = doc_word_freq.get(doc_id, {})
        doc_length = doc_lengths.get(doc_id, 0)
        score = calculate_query_probability(query, term_frequencies, doc_length, total_corpus_frequencies, total_corpus_length, smoothing_lambda)
        scores.append((doc_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    # return scores[:k]
    return scores

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


# Example usage
if __name__ == "__main__":
    # Read data
    queries = read_queries('pythonCode/processedQueries/test_queries.txt')
    documents = read_documents('pythonCode/output/doc_word_list.txt')

    # Generate document word frequency and length
    doc_word_freq, doc_lengths = generate_document_word_frequency_and_length(documents)

# def write_language_model_to_file(doc_word_freq, doc_lengths, output_file):
#     with open(output_file, "w") as f:
#         f.write("Document Term Frequencies:\n")
#         for doc_id, term_freq in doc_word_freq.items():
#             f.write(f"Document {doc_id}:\n")
#             for word, freq in term_freq.items():
#                 f.write(f"\t{word}: {freq}\n")
        
#         f.write("\nDocument Lengths:\n")
#         for doc_id, length in doc_lengths.items():
#             f.write(f"Document {doc_id}: {length}\n")

# # Example usage
# language_model_output_file = "/Users/yspsandeep/Documents/SEM3-2/IR/IR_A2-master/pythonCode/output/language_model.txt"
# write_language_model_to_file(doc_word_freq, doc_lengths, language_model_output_file)



    # Total corpus frequencies (word -> frequency)
    total_corpus_frequencies = Counter()
    for _, content in documents:
        tokens = preprocess(content)
        total_corpus_frequencies.update(tokens)
    total_corpus_length = sum(total_corpus_frequencies.values())

    # Smoothing parameter lambda
    smoothing_lambda = 0.5

    # Retrieve top k documents for each query
    k = 5
def write_results_to_file(queries, documents, doc_word_freq, doc_lengths, total_corpus_frequencies, total_corpus_length, smoothing_lambda, k, output_file):
    with open(output_file, "w") as f:
        for query_id, query in queries:
            top_documents = retrieve_documents(query, documents, doc_word_freq, doc_lengths, total_corpus_frequencies, total_corpus_length, smoothing_lambda, k)
            ndcgScore = calculate_ndcg_for_ranking(top_documents, query_id, k)
            # print(top_documents)
            f.write(f"Query: {query_id} - {query} - {ndcgScore}\n")
            for rank, (doc_id, score) in enumerate(top_documents[:k], 1):
                f.write(f"Rank {rank}: Document {doc_id} - Score: {score}\n")
            f.write("\n")

# Example usage
# top_documents = retrieve_documents(queries[0][1], documents, doc_word_freq, doc_lengths, total_corpus_frequencies, total_corpus_length, smoothing_lambda, k)
# print("HI")
# print(top_documents)
output_file = "pythonCode/output/output_q4_1.txt"  # Specify the file path
write_results_to_file(queries, documents, doc_word_freq, doc_lengths, total_corpus_frequencies, total_corpus_length, smoothing_lambda, k, output_file)



