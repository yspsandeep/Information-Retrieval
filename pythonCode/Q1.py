import os
import string
from nltk.tokenize import word_tokenize
from collections import defaultdict

def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords_list = file.read().splitlines()
    return set(stopwords_list)


def preprocess_text(text, stopwords_set):
    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and filter out non-alphanumeric tokens
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords_set and token.isalpha()]
    return ' '.join(tokens)


def create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def preprocess_and_save(data_folder, stopwords_set):
    processed_folder = os.path.join(os.getcwd(), 'pythonCode', 'processedData')
    create_output_folder(processed_folder)
    processed_data_path = os.path.join(processed_folder, 'processedData.txt')
    with open(processed_data_path, 'w', encoding='utf-8') as processed_file:
        total_files = len(os.listdir(data_folder))
        for i, filename in enumerate(os.listdir(data_folder), start=1):
            file_path = os.path.join(data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:  # Ensure there are at least 4 fields in the line
                        doc_id, title, content = parts[0], parts[2], parts[3]
                        processed_title = preprocess_text(title, stopwords_set)
                        processed_content = preprocess_text(content, stopwords_set)
                        processed_line = f"{doc_id}\t{processed_title}\t{processed_content}\n"
                        processed_file.write(processed_line)
                    else:
                        print(f"Skipping line {line.strip()} in file {filename} due to insufficient fields.")
            print(f"Processed file {i} of {total_files}")

def index_text(text, index, doc_id):
    words = text.split()
    for word in set(words):
        index[word][0] += 1  # Increase document frequency
        index[word][1].add(doc_id)  # Add document ID to the set
        index[word][2][doc_id] += words.count(word)  # Calculate term frequency

def save_index(index, filename, output_folder):
    sorted_index = sorted(index.items(), key=lambda x: x[0])  # Sort index by word
    with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as file:
        for word, info in sorted_index:
            file.write(f"{word}\t{info[0]}\t{' '.join(info[1])}\n")

def save_term_frequency(index, filename, output_folder):
    with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as file:
        for word, info in sorted(index.items()):  # Sort index by word
            for doc_id, tf in sorted(info[2].items()):  # Sort term frequency by doc_id
                file.write(f"{word}\t{doc_id}\t{tf}\n")

def save_cumulative_term_frequency(index, filename, output_folder):
    cumulative_term_frequency = defaultdict(int)
    for word, info in index.items():
        for doc_id, tf in info[2].items():
            cumulative_term_frequency[word] += tf
    sorted_ctf = sorted(cumulative_term_frequency.items(), key=lambda x: x[0])  # Sort by word
    with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as file:
        for word, freq in sorted_ctf:
            file.write(f"{word}\t{freq}\n")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_directory, '..', 'rawdata')
    stopwords_path = os.path.join(current_directory, '..', 'stopWords', 'stopwords.large')
    output_folder = create_output_folder(os.path.join(current_directory, 'output'))

    stopwords_set = load_stopwords(stopwords_path)

    preprocess_and_save(data_folder, stopwords_set)

    index_title = defaultdict(lambda: [0, set(), defaultdict(int)])
    index_content = defaultdict(lambda: [0, set(), defaultdict(int)])
    index_combined = defaultdict(lambda: [0, set(), defaultdict(int)])

    processed_folder = os.path.join(current_directory, 'processedData')
    processed_data_path = os.path.join(processed_folder, 'processedData.txt')
    with open(processed_data_path, 'r', encoding='utf-8') as processed_file:
        for line in processed_file:
            parts = line.strip().split('\t')
            doc_id, title, content = parts[0], parts[1], parts[2]
            index_text(title, index_title, doc_id)
            index_text(content, index_content, doc_id)
            index_text(title + ' ' + content, index_combined, doc_id)

    # Save indexes
    save_index(index_title, 'indexTitle.txt', output_folder)
    save_index(index_content, 'indexContent.txt', output_folder)
    save_index(index_combined, 'indexCombined.txt', output_folder)

    # Save term frequency and cumulative term frequency
    save_term_frequency(index_combined, 'termFrequency.txt', output_folder)
    save_cumulative_term_frequency(index_combined, 'cumulativeTermFrequency.txt', output_folder)

    print("Indexing completed.")
