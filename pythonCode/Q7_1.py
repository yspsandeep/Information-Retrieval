import math
import os
import numpy as np
from sklearn.linear_model import LinearRegression

def load_relevance_data_ndcg(file_path):
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

relevance_folder_ndcg = os.path.join('relevance')
relevance_file_path_ndcg = os.path.join(relevance_folder_ndcg, 'merged.qrel')
relevance_data_ndcg = load_relevance_data_ndcg(relevance_file_path_ndcg)
def calculate_ndcg_for_ranking(myRanking, query_id, k):
    idealRanking  = relevance_data_ndcg[query_id]
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


class FeatureVectorLoader:
    def __init__(self, file_path):
        self.feature_vectors = self.load_feature_vectors(file_path)

    def load_feature_vectors(self, file_path):
        feature_vectors = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                docid = parts[0]
                scores = {}
                for item in parts[1:]:
                    word_score_pairs = item.split()
                    for word_score_pair in word_score_pairs:
                        word, score = word_score_pair.split(':')
                        scores[word] = float(score)
                feature_vectors[docid] = scores
        return feature_vectors

    def get_scores(self, docid):
        if docid in self.feature_vectors:
            return list(self.feature_vectors[docid].values())
        else:
            return None

    def get_word_scores(self, docid):
        if docid in self.feature_vectors:
            return {word: score for word, score in self.feature_vectors[docid].items()}
        else:
            return None

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

    
pathDocVector = os.path.join('pythonCode', 'output', 'Q7_feature_vectors.txt')
pathTrainingVector = os.path.join('pythonCode', 'output', 'Q7_training_feature_vectors.txt')
pathTestVector = os.path.join('pythonCode', 'output', 'Q7_test_feature_vectors.txt')
relevance_folder = os.path.join('relevance')
relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')
document_loader = FeatureVectorLoader(pathDocVector)
training_loader = FeatureVectorLoader(pathTrainingVector)
test_loader = FeatureVectorLoader(pathTestVector)
relevance_data = load_relevance_data(relevance_file_path)

# test_word_scores = test_loader.get_word_scores("PLAIN-2")
# print(test_word_scores)

# Prepare Training Data
X_train = []  # Feature vectors for training
y_train = []  # Relevance scores for training

for query_id, query_vectors in training_loader.feature_vectors.items():
    if query_id in relevance_data:
        for doc_id, relevance_score in relevance_data[query_id]:
            if doc_id in document_loader.feature_vectors:
                doc_features = document_loader.get_scores(doc_id)
                if doc_features:
                    X_train.append(list(query_vectors.values()) + doc_features)
                    y_train.append(relevance_score)

# Train Linear Regression Model
print("Training Linear Regression Model...")
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Linear Regression Model trained successfully.")

# Generate Predictions and Rank Documents
top_n = 10  # Number of top documents to retrieve

# Open file for writing results
results_file_path = os.path.join('pythonCode', 'output', 'Q7_1_results.txt')
with open(results_file_path, 'w') as results_file:
    for query_id, query_vectors in test_loader.feature_vectors.items():
        if query_id in relevance_data:
            results_file.write(f"Query: {query_id}\n")
            query_features = [list(query_vectors.values())] * len(relevance_data[query_id])
            doc_ids = [doc_id for doc_id, _ in relevance_data[query_id]]
            doc_features = [document_loader.get_scores(doc_id) for doc_id in doc_ids]
            X_test = [query_feat + doc_feat if doc_feat else query_feat for query_feat, doc_feat in zip(query_features, doc_features)]
            predictions = regressor.predict(X_test)
            sorted_docs = sorted(zip(doc_ids, predictions), key=lambda x: x[1], reverse=True)
            sorted_docs_topK = sorted(zip(doc_ids, predictions), key=lambda x: x[1], reverse=True)[:top_n]
            ndcgscore = calculate_ndcg_for_ranking(sorted_docs, query_id, k=10)
            results_file.write(f"NDCG: {ndcgscore}\n")
            results_file.write("Top Documents:\n")
            for rank, (doc_id, score) in enumerate(sorted_docs_topK, 1):
                results_file.write(f"Rank {rank}: Document ID: {doc_id}, Predicted Relevance Score: {score}\n")
            results_file.write("\n")

print("Results saved successfully.")