# query_pipeline.py
# Final version with improved prompting, re-added logging, and better error handling.

from typing import List, Dict, Any
from embedding_manager import EmbeddingManager
from vectordb import FAISSManager
from postgres_manager import PostgresManager
import google.generativeai as genai
from settings import settings
import logging

logger = logging.getLogger(__name__)

class QueryPipeline:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_db = FAISSManager(self.embedding_manager.get_dimension())
        self.postgres_db = PostgresManager()
        # Configure Gemini API key
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
        else:
            logger.error("GEMINI_API_KEY not found. Query generation will fail.")

    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a single query through the RAG pipeline."""
        try:
            # --- CRITICAL FIX ---
            # Reload the index from disk to ensure it reflects any newly added documents.
            logger.info("Reloading FAISS index to ensure data is current.")
            self.vector_db.load_or_create_index()
            # -----------------

            if self.vector_db.get_total_vectors() == 0:
                logger.warning("Query attempted on an empty vector index.")
                return {"query": query, "answer": "The document knowledge base is empty. Please upload a document first.", "sources": []}

            # Step 1: Generate an embedding for the user's query.
            query_embedding = self.embedding_manager.generate_query_embedding(query)

            # Step 2: Perform a similarity search in the vector database.
            scores, vector_ids = self.vector_db.search(query_embedding, k=top_k)

            if not vector_ids or (len(vector_ids) > 0 and vector_ids[0] == -1):
                logger.warning(f"Query '{query[:50]}...' found no relevant chunks.")
                return {
                    "query": query,
                    "answer": "I could not find any relevant information in the document for your question.",
                    "sources": []
                }

            # Step 3: Retrieve the actual text chunks from PostgreSQL using their IDs.
            valid_vector_ids = [int(vid) for vid in vector_ids if vid != -1]
            chunks = self.postgres_db.get_chunks_by_vector_ids(valid_vector_ids)

            if not chunks:
                logger.warning(f"Vector search found IDs {valid_vector_ids}, but no content was retrieved from PostgreSQL.")
                return {
                    "query": query,
                    "answer": "Found potential matches but could not retrieve the document content. The database may be out of sync.",
                    "sources": []
                }

            # Step 4: Prepare the context for the language model.
            context = self._prepare_context(chunks, scores[:len(chunks)])

            # Step 5: Generate a human-readable answer using the LLM.
            answer = self._generate_answer(query, context)

            # Step 6: Format the sources for the final response.
            sources = [
                {
                    "filename": chunk.get("filename", "N/A"),
                    "content": chunk["content"][:250] + "..." if len(chunk["content"]) > 250 else chunk["content"],
                    "score": float(score)
                }
                for chunk, score in zip(chunks, scores[:len(chunks)])
            ]

            self.postgres_db.store_query_result(query, sources)

            return {
                "query": query,
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Critical error during query processing for '{query[:50]}...': {e}", exc_info=True)
            return {
                "query": query,
                "answer": "An unexpected server error occurred while processing your request.",
                "sources": [],
                "error": str(e)
            }
            
    def process_multiple_queries(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries"""
        results = []
        for query in queries:
            result = self.process_query(query, top_k)
            results.append(result)
        return results


    def _prepare_context(self, chunks: List[Dict], scores: List[float]) -> str:
        """Combine retrieved chunks into a single string for the LLM context."""
        context_parts = []
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            context_parts.append(
                f"--- Document Snippet {i+1} (Relevance Score: {score:.4f}) ---\n{chunk['content'].strip()}"
            )
        return "\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Sends the query and context to the Gemini LLM for a final answer."""
        if not settings.GEMINI_API_KEY:
            return "Cannot generate answer: Gemini API key is not configured."

        try:
            prompt = f"""
You are an expert assistant for analyzing insurance policy documents. Your task is to answer the user's question based ONLY on the provided document snippets.

**Instructions:**
1.  Read the user's QUESTION carefully.
2.  Review the DOCUMENT SNIPPETS provided below.
3.  Synthesize a clear and direct answer using only the information from the snippets.
4.  If the answer is explicitly stated, quote the relevant part.
5.  If the information is not available in the snippets, you MUST state: "Based on the provided document sections, the information is not available." Do not use any external knowledge or make assumptions.

---
**DOCUMENT SNIPPETS:**

{context}
---

**QUESTION:**
{query}
---

**ANSWER:**
"""

            # As requested by the user, print the prompt for debugging.
            print("=== Gemini Prompt ===")
            print(prompt)
            print("=====================")

            model = genai.GenerativeModel(model_name=settings.LLM_MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=500  # Increased to allow for more complete answers
                )
            )

            # Safely get the text from the response.
            answer = getattr(response, "text", "").strip()

            if not answer:
                logger.warning("Gemini model returned an empty response.")
                return "The model did not generate a response for this query."
            
            return answer

        except Exception as e:
            logger.error(f"Error during Gemini API call: {str(e)}", exc_info=True)
            return "An error occurred while communicating with the language model."
