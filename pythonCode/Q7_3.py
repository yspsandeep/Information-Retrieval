import math
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# Your existing code...
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

def load_relevance_data(file_path, docid_list):
    relevance_data = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            doc_id = parts[2]
            relevance_score = int(parts[3])
            if query_id not in relevance_data:
                # Initialize relevance scores for all documents to 0
                relevance_data[query_id] = {doc: 0 for doc in docid_list}
            # Update relevance score if available
            relevance_data[query_id][doc_id] = relevance_score
    return relevance_data

trainingQuery_id = []
testQuery_id = []

def read_queries_from_file_training(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            trainingQuery_id.append(query_id)

def read_queries_from_file_test(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            testQuery_id.append(query_id)

processed_queries_folder = os.path.join(os.getcwd(),'pythonCode', 'processedQueries')
files_test = ['combined_test_queries.txt']
files_training = ['combined_training_queries.txt']

for file_name in files_training:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file_training(file_path)
    else:
        print(f"File not found: {file_path}")

for file_name in files_test:
    file_path = os.path.join(processed_queries_folder, file_name)
    if os.path.exists(file_path):
        read_queries_from_file_test(file_path)
    else:
        print(f"File not found: {file_path}")

pathDocVector = os.path.join('pythonCode', 'output', 'Q7_feature_vectors.txt')
pathTrainingVector = os.path.join('pythonCode', 'output', 'Q7_training_feature_vectors.txt')
pathTestVector = os.path.join('pythonCode', 'output', 'Q7_test_feature_vectors.txt')
relevance_folder = os.path.join('relevance')
relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')
document_loader = FeatureVectorLoader(pathDocVector)
training_loader = FeatureVectorLoader(pathTrainingVector)
test_loader = FeatureVectorLoader(pathTestVector)
relevance_data = load_relevance_data(relevance_file_path,docid_list)

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


class ListwiseRanker:
    def __init__(self, document_loader, training_loader, relevance_data):
        self.document_loader = document_loader
        self.training_loader = training_loader
        self.relevance_data = relevance_data
        self.ranker = GradientBoostingRegressor()


    def train(self, k):
        X_train = []
        y_train = []
        count = 0
        i = 0
        for query_id in trainingQuery_id:
            print("currently processing query: ", i)
            i += 1
            query_vector = self.training_loader.get_scores(query_id)
            if query_vector is not None:
                for doc_id, relevance_score in self.relevance_data[query_id].items():
                    doc_vector = self.document_loader.get_scores(doc_id)
                    if doc_vector is not None:
                        X_train.append(np.subtract(query_vector, doc_vector))
                        y_train.append(relevance_score)
            count += 1
            if count >= k:
                break
        self.ranker.fit(X_train, y_train)

    def rank(self, query_loader):
        rankings = {}
        for query_id in testQuery_id:
            query_vector = query_loader.get_scores(query_id)
            if query_vector is not None:
                scores = {}
                for doc_id in docid_list:
                    doc_vector = self.document_loader.get_scores(doc_id)
                    if doc_vector is not None:
                        score = self.ranker.predict([np.subtract(query_vector, doc_vector)])
                        scores[doc_id] = score
                # rankings[query_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                rankings[query_id] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return rankings

listwise_ranker = ListwiseRanker(document_loader, training_loader, relevance_data)
listwise_ranker.train(k=100)
rankings = listwise_ranker.rank(test_loader)

# Write rankings to a text file
output_file = os.path.join('pythonCode', 'output', 'Q7_3_results.txt')
with open(output_file, 'w') as f:
    for query_id, docs in rankings.items():
        k = 10
        ndcg = calculate_ndcg_for_ranking(docs, query_id, k)
        f.write(f"Query: {query_id}\n")
        f.write(f"ndcg: {ndcg}\n")
        for rank, (doc_id, score) in enumerate(docs, 1):
            f.write(f"{rank}. Doc ID: {doc_id}, Score: {score}\n")
        f.write('\n')
