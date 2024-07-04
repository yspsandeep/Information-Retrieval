import os

# Path to the processedQueries folder
processed_queries_folder = os.path.join('pythonCode', 'processedQueries')

# Function to concatenate queries with the same query ID
def concatenate_queries(file_path):
    combined_queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t', 1)
            if query_id in combined_queries:
                combined_queries[query_id] += ' ' + query_text
            else:
                combined_queries[query_id] = query_text
    return combined_queries

# Function to write combined queries to file
def write_combined_queries(combined_queries, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for query_id, query_text in combined_queries.items():
            file.write(f"{query_id}\t{query_text}\n")

# Process training queries
training_queries_file = os.path.join(processed_queries_folder, 'training_queries.txt')
training_combined_queries = concatenate_queries(training_queries_file)
output_training_file = os.path.join(processed_queries_folder, 'combined_training_queries.txt')
write_combined_queries(training_combined_queries, output_training_file)

# Process test queries
test_queries_file = os.path.join(processed_queries_folder, 'test_queries.txt')
test_combined_queries = concatenate_queries(test_queries_file)
output_test_file = os.path.join(processed_queries_folder, 'combined_test_queries.txt')
write_combined_queries(test_combined_queries, output_test_file)

# Process dev queries
dev_queries_file = os.path.join(processed_queries_folder, 'dev_queries.txt')
dev_combined_queries = concatenate_queries(dev_queries_file)
output_dev_file = os.path.join(processed_queries_folder, 'combined_dev_queries.txt')
write_combined_queries(dev_combined_queries, output_dev_file)

print("Combined queries have been saved to:")
print("- Training queries:", output_training_file)
print("- Test queries:", output_test_file)
print("- Dev queries:", output_dev_file)
