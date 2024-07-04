import os
import re


training_query_files = ["train.nontopic-titles.queries", "train.titles.queries", "train.vid-desc.queries","train.vid-titles.queries"]
test_query_files = ["test.nontopic-titles.queries", "test.titles.queries", "test.vid-desc.queries","test.vid-titles.queries"]
dev_query_files = ["dev.nontopic-titles.queries", "dev.titles.queries", "dev.vid-desc.queries","dev.vid-titles.queries"]


queries_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'queries')
processed_queries_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processedQueries')

if not os.path.exists(processed_queries_folder):
    os.makedirs(processed_queries_folder)

def clean_query(query):
    query_id, query_text = query.split('\t', 1)
    query_text = re.sub(r'[^\w\s]', '', query_text)  # Remove punctuation
    return f"{query_id}\t{query_text.strip()}"  # Ensure query text is stripped of leading/trailing whitespace

def combine_queries(query_files):
    all_queries = set()  # Using a set to automatically remove duplicates
    for filename in query_files:
        file_path = os.path.join(queries_folder, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                queries = file.readlines()
                for query in queries:
                    cleaned_query = clean_query(query)
                    all_queries.add(cleaned_query.strip())  # Remove leading/trailing whitespace
    # Sort queries by ID
    sorted_queries = sorted(all_queries, key=lambda x: int(x.split('\t')[0].split('-')[-1]))
    return sorted_queries

training_queries = combine_queries(training_query_files)
test_queries = combine_queries(test_query_files)
dev_queries = combine_queries(dev_query_files)

output_training_file = os.path.join(processed_queries_folder, 'training_queries.txt')
output_test_file = os.path.join(processed_queries_folder, 'test_queries.txt')
output_dev_file = os.path.join(processed_queries_folder, 'dev_queries.txt')

with open(output_training_file, 'w', encoding='utf-8') as file:
    file.write('\n'.join(training_queries))

with open(output_test_file, 'w', encoding='utf-8') as file:
    file.write('\n'.join(test_queries))

with open(output_dev_file, 'w', encoding='utf-8') as file:
    file.write('\n'.join(dev_queries))

print("Combined queries have been saved to:")
print("- Training queries:", output_training_file)
print("- Test queries:", output_test_file)
print("- Dev queries:", output_dev_file)
