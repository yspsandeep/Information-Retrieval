import os
from math import log
from collections import defaultdict

def load_term_frequency(term_frequency_file):
    term_frequency = defaultdict(dict)
    with open(term_frequency_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, doc_id, tf = line.strip().split('\t')
            term_frequency[word][doc_id] = int(tf)
    return term_frequency

def load_index_combined(index_combined_file):
    index_combined = defaultdict(lambda: [0, set()])
    with open(index_combined_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, df, doc_ids = line.strip().split('\t')
            index_combined[word][0] = int(df)
            index_combined[word][1] = set(doc_ids.split())
    return index_combined

def calculate_tf_idf(tf, df, corpus_size):
    return (1 + log(tf)) * log(corpus_size / df)

def generate_document_vector(term_frequency, index_combined, doc_ids):
    document_vectors = {}
    corpus_size = len(doc_ids)  # Total number of documents in the corpus
    total_docs = len(doc_ids)  # Total number of documents
    done_docs = 0
    for doc_id in doc_ids:
        for word, tf in term_frequency.items():
            if doc_id in tf and word in index_combined:
                # df = index_combined[word][0]
                # tf_idf = calculate_tf_idf(tf[doc_id], df, corpus_size)
                if doc_id not in document_vectors:
                    document_vectors[doc_id] = defaultdict(float)
                # document_vectors[doc_id][word] = tf_idf
                document_vectors[doc_id][word] = 1
        done_docs += 1
        print(f"Processed {done_docs} documents out of {total_docs}")
    return document_vectors

def save_document_vectors(document_vectors, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        sorted_doc_ids = sorted(document_vectors.keys())  # Sort document IDs
        for doc_id in sorted_doc_ids:
            vector = document_vectors[doc_id]
            components = ' '.join([f"{word}:{value}" for word, value in sorted(vector.items())])
            file.write(f"{doc_id}\t{components}\n")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(current_directory, 'vectors')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    term_frequency_file = os.path.join(current_directory, 'output', 'termFrequency.txt')
    index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
    final_data_file = os.path.join(current_directory, 'processedData', 'processedData.txt')
    output_file = os.path.join(output_folder, 'docVectors.tsv')

    term_frequency = load_term_frequency(term_frequency_file)
    index_combined = load_index_combined(index_combined_file)

    # Read the finalData.txt file to get the list of document IDs
    doc_ids = set()
    with open(final_data_file, 'r', encoding='utf-8') as file:
        for line in file:
            doc_id = line.strip().split('\t')[0]
            doc_ids.add(doc_id)

    document_vectors = generate_document_vector(term_frequency, index_combined, doc_ids)
    save_document_vectors(document_vectors, output_file)

    print("Document vectors generated and saved successfully.")
