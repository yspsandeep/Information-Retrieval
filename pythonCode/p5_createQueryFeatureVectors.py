from collections import defaultdict
import os


def load_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_id = parts[0]
            query_text = parts[1]
            queries[query_id] = query_text
    return queries

test_queries_file_path = os.path.join('pythonCode', 'processedQueries', 'combined_test_queries.txt')
test_queries = load_queries(test_queries_file_path)

# Load training queries
training_queries_file_path = os.path.join('pythonCode', 'processedQueries', 'combined_training_queries.txt')
training_queries = load_queries(training_queries_file_path)

# it gives word and corresponding docids
def load_index_combined(index_combined_file):
    index_combined = defaultdict(lambda: [0, set()])
    with open(index_combined_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, df, doc_ids = line.strip().split('\t')
            index_combined[word][0] = int(df)
            index_combined[word][1] = set(doc_ids.split())
    return index_combined

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

def check_word_in_document(word, doc_id, index_map):
    if word in index_map and doc_id in index_map[word]:
        return 1
    return 0


# Read the file and extract term frequencies
with open("pythonCode/output/cumulativeTermFrequency.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
# Split lines into (word, frequency) pairs and convert frequency to integer
word_freq_pairs = []
for line in lines:
    parts = line.strip().split('\t')
    if len(parts) == 2:  # Ensure the line has exactly 2 tab-separated values
        word = parts[0]
        freq = int(parts[1])
        word_freq_pairs.append((word, freq))

sorted_word_freq_pairs = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)

# Extract the top k words and sort alphabetically
k = 100
top_k_words = sorted([pair[0] for pair in sorted_word_freq_pairs[:k]])

with open("pythonCode/output/Q7_training_feature_vectors.txt", "w", encoding="utf-8") as output_file:
    for query_id, query_text in training_queries.items():
        scores = {}
        for word in top_k_words:
            scores[word] = 1 if word in query_text else 0

        output_line = f"{query_id}\t{' '.join([f'{word}:{score}' for word, score in scores.items()])}\n"
        output_file.write(output_line)

with open("pythonCode/output/Q7_test_feature_vectors.txt", "w", encoding="utf-8") as output_file:
    for query_id, query_text in test_queries.items():
        scores = {}
        for word in top_k_words:
            scores[word] = 1 if word in query_text else 0

        output_line = f"{query_id}\t{' '.join([f'{word}:{score}' for word, score in scores.items()])}\n"
        output_file.write(output_line)



