from collections import defaultdict
from math import log
import os

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

def load_term_frequency(term_frequency_file):
    term_frequency = defaultdict(dict)
    with open(term_frequency_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, doc_id, tf = line.strip().split('\t')
            term_frequency[word][doc_id] = int(tf)
    return term_frequency

def calculate_tf_idf(tf, df, corpus_size):
    return (1 + log(tf)) * log(corpus_size / df)

# searches whether a word exists in a doc
def check_word_in_document(word, doc_id, index_map):
    if word in index_map and doc_id in index_map[word]:
        return 1
    return 0

# Sort the pairs by frequency in descending order
sorted_word_freq_pairs = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)

# Extract the top k words and sort alphabetically
k = 100
top_k_words = sorted([pair[0] for pair in sorted_word_freq_pairs[:k]])

# Build index map for quick lookup
current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
index_combined = load_index_combined(index_combined_file)
index_map = build_index_map(index_combined_file)
tf = load_term_frequency("pythonCode/output/termFrequency.txt")
corpus_size = 5371

# Now, let's create binary feature vectors for each document
# Read the document word lists and create feature vectors
feature_vectors = {}

with open("pythonCode/output/doc_word_list.txt", "r", encoding="utf-8") as doc_file:
    for line in doc_file:
        parts = line.strip().split('\t')
        doc_id = parts[0]
        words_in_doc = parts[1:]

        # Initialize the feature vector for this document
        feature_vector = [0] * k

        # Set binary values for words present in the document
        for i, word in enumerate(top_k_words):
            if(check_word_in_document(word, doc_id, index_map) == 1):
                # feature_vector[i] = 1
                df = index_combined[word][0]
                tf_idf = calculate_tf_idf(tf[word][doc_id], df, corpus_size)
                feature_vector[i] = tf_idf

        # Store the feature vector for this document
        feature_vectors[doc_id] = feature_vector

# Write feature vectors to a text file
with open("pythonCode/output/Q7_feature_vectors.txt", "w", encoding="utf-8") as output_file:
    for doc_id, vector in feature_vectors.items():
        word_value_pairs = [f"{word}:{value}" for word, value in zip(top_k_words, vector)]
        vector_str = ' '.join(word_value_pairs)
        output_file.write(f"{doc_id}\t{vector_str}\n")


# print(top_k_words)
