import os

current_directory = os.path.dirname(os.path.abspath(__file__))
index_combined_file = os.path.join(current_directory, 'output', 'indexCombined.txt')

def generate_doc_word_file(index_combined_file, output_file):
    doc_words = {}
    with open(index_combined_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, docfreq, docs = line.strip().split('\t')
            doc_ids = docs.split()
            for doc_id in doc_ids:
                if doc_id not in doc_words:
                    doc_words[doc_id] = []
                doc_words[doc_id].append(word)

    with open(output_file, 'w', encoding='utf-8') as output:
        sorted_doc_ids = sorted(doc_words.keys())  # Sort document IDs lexicographically
        for doc_id in sorted_doc_ids:
            words = doc_words[doc_id]
            output.write(f"{doc_id}\t{' '.join(words)}\n")

# Output file path
output_file = os.path.join(current_directory, 'output', 'doc_word_list.txt')

# Generate the document-word file
generate_doc_word_file(index_combined_file, output_file)

print("Document-word file generated and sorted lexicographically.")
