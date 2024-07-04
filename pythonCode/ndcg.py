import os
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

def calculate_ndcg_for_ranking(myRanking, query_id, k):
    relevance_folder = os.path.join('relevance')
    relevance_file_path = os.path.join(relevance_folder, 'merged.qrel')
    relevance_data = load_relevance_data(relevance_file_path)
    idealRanking  = relevance_data[query_id]
    if(len(idealRanking)<k):
        print("Error: idealRanking does not have enough documents.")
        return
    idealTopK, myTopK = sort_and_select_top_k(myRanking, idealRanking, k)
    ndcg_score = ndcg_at_k(myTopK, idealTopK, k)
    if ndcg_score > 1:
        print("Error: idealRanking does not have enough non-zero relevance documents.")
        return
    print("NDCG Score:", ndcg_score)
