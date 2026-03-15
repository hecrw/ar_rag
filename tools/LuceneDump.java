import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PrefixQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;

/**
 * Dumps stored fields from a Lucene index as JSON Lines.
 *
 * Usage: java LuceneDump <index-path> [options]
 *   --fields        : print field names from first doc, then exit
 *   --count         : print total document count, then exit
 *   --limit N       : max docs to dump (default: all)
 *   --offset N      : skip first N docs (default: 0)
 *   --prefix FIELD VALUE : filter docs where FIELD starts with VALUE
 *   --batch FIELD   : read prefixes from stdin (one per line), search for each
 */
public class LuceneDump {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java LuceneDump <index-path> [options]");
            System.exit(1);
        }

        String indexPath = args[0];
        boolean fieldsOnly = false;
        boolean countOnly = false;
        int limit = -1;
        int offset = 0;
        String prefixField = null;
        String prefixValue = null;
        String batchField = null;

        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--fields": fieldsOnly = true; break;
                case "--count": countOnly = true; break;
                case "--limit": limit = Integer.parseInt(args[++i]); break;
                case "--offset": offset = Integer.parseInt(args[++i]); break;
                case "--prefix":
                    prefixField = args[++i];
                    prefixValue = args[++i];
                    break;
                case "--batch":
                    batchField = args[++i];
                    break;
            }
        }

        PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
        FSDirectory dir = FSDirectory.open(Paths.get(indexPath));
        IndexReader reader = DirectoryReader.open(dir);
        int numDocs = reader.maxDoc();

        if (countOnly) {
            out.println(numDocs);
            reader.close();
            return;
        }

        if (fieldsOnly) {
            if (numDocs > 0) {
                Document doc = reader.document(0);
                out.print("[");
                boolean first = true;
                for (IndexableField field : doc.getFields()) {
                    if (!first) out.print(",");
                    out.print("\"" + escapeJson(field.name()) + "\"");
                    first = false;
                }
                out.println("]");
            }
            reader.close();
            return;
        }

        // Batch mode: read prefixes from stdin, output docs for each
        if (batchField != null) {
            IndexSearcher searcher = new IndexSearcher(reader);
            BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            String line;
            while ((line = stdin.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                PrefixQuery query = new PrefixQuery(new Term(batchField, line));
                TopDocs results = searcher.search(query, numDocs);
                // Print separator with prefix so Python knows which book
                out.println("###BOOK:" + line);
                for (ScoreDoc sd : results.scoreDocs) {
                    Document doc = reader.document(sd.doc);
                    printDoc(out, doc);
                }
            }
            reader.close();
            return;
        }

        // Single prefix search
        if (prefixField != null) {
            IndexSearcher searcher = new IndexSearcher(reader);
            PrefixQuery query = new PrefixQuery(new Term(prefixField, prefixValue));
            int maxResults = limit > 0 ? limit : numDocs;
            TopDocs results = searcher.search(query, maxResults);
            for (ScoreDoc sd : results.scoreDocs) {
                Document doc = reader.document(sd.doc);
                printDoc(out, doc);
            }
            reader.close();
            return;
        }

        // Sequential dump
        int count = 0;
        for (int i = offset; i < numDocs; i++) {
            if (limit >= 0 && count >= limit) break;
            Document doc = reader.document(i);
            printDoc(out, doc);
            count++;
        }

        reader.close();
    }

    private static void printDoc(PrintStream out, Document doc) {
        out.print("{");
        boolean first = true;
        for (IndexableField field : doc.getFields()) {
            if (!first) out.print(",");
            String value = field.stringValue();
            if (value == null) {
                Number num = field.numericValue();
                value = num != null ? num.toString() : "";
            }
            out.print("\"" + escapeJson(field.name()) + "\":\"" + escapeJson(value) + "\"");
            first = false;
        }
        out.println("}");
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder(s.length());
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int)c));
                    else sb.append(c);
            }
        }
        return sb.toString();
    }
}
