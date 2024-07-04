import java.io.IOException;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

public class LuceneTester {

    String indexDir = "./index";
    String dataDir = "./data";
    Indexer indexer;
    Searcher searcher;

    public static void main(String[] args) {
        LuceneTester tester;
        try {
            tester = new LuceneTester();
            // tester.createIndex();
            tester.search("deep");
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
    }

    private void createIndex() throws IOException {
        indexer = new Indexer(indexDir);
        int numIndexed;
        long startTime = System.currentTimeMillis();
        numIndexed = indexer.createIndex(dataDir, new TextFileFilter());
        long endTime = System.currentTimeMillis();
        indexer.close();
        System.out.println(numIndexed + " File indexed, time taken: " + (endTime - startTime) + " ms");
    }

    private void search(String searchQuery) throws IOException, ParseException {
        searcher = new Searcher(indexDir);
        TopDocs hits = searcher.searchExactTerms(searchQuery);
      
        for (ScoreDoc scoreDoc : hits.scoreDocs) {
            Document doc = searcher.getDocument(scoreDoc);
            String title = doc.get("title");
            String content = doc.get("contents");
            String docID = doc.get("filename");
          
            boolean searchTermInTitle = title != null && title.toLowerCase().contains(searchQuery.toLowerCase());
            boolean searchTermInContent = content != null && content.toLowerCase().contains(searchQuery.toLowerCase());
          
            System.out.println("DocID: " + docID + ", Search term found in title: " + searchTermInTitle + ", Search term found in content: " + searchTermInContent);
        }
      
        searcher.close();
    }
}
