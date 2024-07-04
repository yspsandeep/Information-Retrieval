import java.io.IOException;

public class Q1 {

    public static void main(String[] args) {
        // Call Remove.createDocuments() before indexing
        Remove.createDocuments("rawdata/doc_dump.txt");
        Remove.createDocuments("rawdata/nfdump.txt");

        // Continue with indexing
        try {
            Indexer indexer = new Indexer("./index"); // Assuming the index will be stored in the "./index" directory
            int numIndexed = indexer.createIndex("./data", new TextFileFilter()); // Index files from the "./data" directory
            indexer.close();
            System.out.println(numIndexed + " files indexed successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
