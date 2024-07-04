import math
import os
import numpy as np
from itertools import combinations
from collections import defaultdict
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
    if not scores:  # Check if scores list is empty
        return []
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

def retrieve_docids_from_file(file_path):
    docid_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            docid = line.strip().split('\t')[0]
            docid_list.append(docid)
    return docid_list

def load_relevance_data(file_path, docid_list):
    relevance_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                relevance_data[query_id] = {doc: 0 for doc in docid_list}
            relevance_data[query_id][doc_id] = relevance_score
    return relevance_data

def read_queries_from_file_training(file_path):
    trainingQuery_id = []
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            trainingQuery_id.append(query_id)
    return trainingQuery_id

def read_queries_from_file_test(file_path):
    testQuery_id = []
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            testQuery_id.append(query_id)
    return testQuery_id


class PairwiseRanking:
    def __init__(self, document_loader, relevance_data):
        self.document_loader = document_loader
        self.relevance_data = relevance_data
        self.model = None

    def calculate_relevance_score(self, doc1, doc2, query_id):
        scores1 = self.document_loader.get_scores(doc1)
        scores2 = self.document_loader.get_scores(doc2)
        relevance_scores = self.relevance_data[query_id]
        relevance_doc1 = relevance_scores.get(doc1, 0)
        relevance_doc2 = relevance_scores.get(doc2, 0)
        return relevance_doc1 - relevance_doc2

    def generate_pairs(self, query_id):
        doc_ids = list(self.relevance_data[query_id].keys())
        pairs = list(combinations(doc_ids, 2))
        return pairs

    def train(self, training_queries, k):
        X = []
        y = []
        count = 0
        i = 0
        for query_id in training_queries:
            print(i)
            i += 1
            if count >= k:
                break
            pairs = self.generate_pairs(query_id)
            for pair in pairs:
                doc1, doc2 = pair
                score = self.calculate_relevance_score(doc1, doc2, query_id)
                X.append([score])
                y.append(1 if score > 0 else -1)
            count += 1
        X = np.array(X)
        y = np.array(y)
        self.model = LinearRegression().fit(X, y)

    def rank_documents(self, test_queries):
        results = defaultdict(list)
        for query_id in test_queries:
            doc_ids = list(self.relevance_data[query_id].keys())
            X = []
            for doc_id in doc_ids:
                score = self.calculate_relevance_score(doc_id, '', query_id)
                X.append([score])
            X = np.array(X)
            predicted_scores = self.model.predict(X)
            sorted_docs = [(doc_id, score) for score, doc_id in sorted(zip(predicted_scores, doc_ids), reverse=True)]
            results[query_id] = sorted_docs
            # results[query_id] = sorted_docs[:10]
        return results



def write_results_to_file(results, output_file):
    with open(output_file, 'w') as file:
        k = 10
        for query_id, docs in results.items():
            ndcgScore = calculate_ndcg_for_ranking(docs, query_id, k)
            docID = [doc_id for doc_id, _ in docs]
            topdocs = docID[:10]
            file.write(f"{query_id}\t{' '.join(topdocs)}\n")
            file.write(f"NDCG Score: {ndcgScore}\n")


# Load data
final_data_file = os.path.join(os.getcwd(), 'pythonCode', 'processedData', 'processedData.txt')
pathDocVector = os.path.join('pythonCode', 'output', 'Q7_feature_vectors.txt')
pathTrainingVector = os.path.join('pythonCode', 'output', 'Q7_training_feature_vectors.txt')
pathTestVector = os.path.join('pythonCode', 'output', 'Q7_test_feature_vectors.txt')
relevance_folder = os.path.join('relevance')
relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')

docid_list = retrieve_docids_from_file(final_data_file)
document_loader = FeatureVectorLoader(pathDocVector)
training_loader = FeatureVectorLoader(pathTrainingVector)
test_loader = FeatureVectorLoader(pathTestVector)
relevance_data = load_relevance_data(relevance_file_path, docid_list)
trainingQuery_id = read_queries_from_file_training(os.path.join('pythonCode', 'processedQueries', 'training_queries.txt'))
testQuery_id = read_queries_from_file_test(os.path.join('pythonCode', 'processedQueries', 'test_queries.txt'))

# Initialize pairwise ranking model
pairwise_ranking = PairwiseRanking(document_loader, relevance_data)

# Train model
pairwise_ranking.train(trainingQuery_id,100)

# Rank documents for test queries
results= pairwise_ranking.rank_documents(testQuery_id)

# Write results to file
output_file = os.path.join('pythonCode', 'output', 'Q7_2_results.txt')
write_results_to_file(results, output_file)
