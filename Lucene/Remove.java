import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Remove {
    public static void main(String[] args) {
        String docDumpFilePath = "rawdata/doc_dump.txt";
        String nfdumpFilePath = "rawdata/nfdump.txt";
        createDocuments(docDumpFilePath);
        createDocuments(nfdumpFilePath);
    }

    public static void createDocuments(String filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] fields = line.split("\t");
                if (fields.length >= 1) {
                    String docId = fields[0];
                    StringBuilder content = new StringBuilder();
                    for (int i = 1; i < fields.length; i++) {
                        content.append(fields[i]).append("\n");
                    }
                    String fileName = "data/" + docId + ".txt";
                    createDocument(fileName, content.toString());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void createDocument(String fileName, String content) {
        try (FileWriter fw = new FileWriter(fileName)) {
            fw.write(content.trim());
            System.out.println("Document " + fileName + " created.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
