import os
from collections import OrderedDict

def load_vocabulary_from_combined_index():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')
    if not os.path.exists(index_combined_file):
        print("Error: indexCombined.txt not found.")
        return set()
    vocabulary = set()
    with open(index_combined_file, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip().split('\t')[0]
            vocabulary.add(word)
    return vocabulary

def retrieve_document_vector_values(doc_id):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'vectors', 'docVectors.tsv')
    vocabulary = load_vocabulary_from_combined_index()
    document_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            current_doc_id = parts[0]
            if current_doc_id == doc_id:
                vector_parts = parts[1].split()
                vector = OrderedDict((word, float(value)) for word, value in (pair.split(':') for pair in vector_parts))
                document_vectors[current_doc_id] = vector
                break

    # Fill in missing values with zeros and preserve order
    vector = OrderedDict()
    if doc_id in document_vectors:
        vector.update(document_vectors[doc_id])
    for word in vocabulary:
        if word not in vector:
            vector[word] = 0.0

    # Sort the vector
    sorted_vector = OrderedDict(sorted(vector.items()))
    return sorted_vector
    # Extract and return only the values
    # values = list(sorted_vector.values())

    # return values

# Example usage
if __name__ == "__main__":
    doc_id = "MED-1"
    vector_values = retrieve_document_vector_values(doc_id)
    # print("Vector values:", vector_values)
    print(len(vector_values))
