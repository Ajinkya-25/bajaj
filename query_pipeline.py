# query_pipeline.py
# Final version with a professionally tuned prompt for high-quality answers.

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
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
        else:
            logger.error("GEMINI_API_KEY not found. Query generation will fail.")

    def process_query(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """Process a single query through the RAG pipeline."""
        try:
            logger.info("Reloading FAISS index to ensure data is current.")
            self.vector_db.load_or_create_index()

            if self.vector_db.get_total_vectors() == 0:
                return {"query": query, "answer": "The document knowledge base is empty. Please upload a document first.", "sources": []}

            query_embedding = self.embedding_manager.generate_query_embedding(query)
            scores, vector_ids = self.vector_db.search(query_embedding, k=top_k)

            if not vector_ids or (len(vector_ids) > 0 and vector_ids[0] == -1):
                return {"query": query, "answer": "I could not find any relevant information in the document for your question.", "sources": []}

            valid_vector_ids = [int(vid) for vid in vector_ids if vid != -1]
            chunks = self.postgres_db.get_chunks_by_vector_ids(valid_vector_ids)

            if not chunks:
                return {"query": query, "answer": "Found potential matches but could not retrieve document content.", "sources": []}

            context = self._prepare_context(chunks, scores[:len(chunks)])
            answer = self._generate_answer(query, context)

            sources = [{"filename": chunk.get("filename", "N/A"), "content": chunk["content"][:250] + "...", "score": float(score)} for chunk, score in zip(chunks, scores[:len(chunks)])]
            self.postgres_db.store_query_result(query, sources)

            return {"query": query, "answer": answer, "sources": sources}
        except Exception as e:
            logger.error(f"Critical error during query processing: {e}", exc_info=True)
            return {"query": query, "answer": "An unexpected server error occurred.", "sources": [], "error": str(e)}

    def process_multiple_queries(self, queries: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        return [self.process_query(q, top_k) for q in queries]

    def _prepare_context(self, chunks: List[Dict], scores: List[float]) -> str:
        """Combines retrieved chunks into a single string for the LLM context."""
        return "\n\n".join([f"--- Document Snippet {i+1} (Relevance Score: {score:.4f}) ---\n{chunk['content'].strip()}" for i, (chunk, score) in enumerate(zip(chunks, scores))])

    def _generate_answer(self, query: str, context: str) -> str:
        """Sends the query and context to the Gemini LLM for a final, high-quality answer."""
        if not settings.GEMINI_API_KEY:
            return "Cannot generate answer: Gemini API key is not configured."

        try:
            # --- FINAL PROMPT STRUCTURE ---
            prompt = f"""
You are a highly skilled insurance policy analyst. Your sole task is to provide a precise, comprehensive, and direct answer to the question based *only* on the provided text snippets from a policy document.

**Instructions:**
1.  **Synthesize a Complete and Concise Answer:** Read all the provided snippets and extract all relevant details to form a complete, yet concise answer. Your answer should be a single, well-structured paragraph that directly addresses all parts of the user's question, including any conditions, limits, or time periods mentioned in the text.
2.  **Be Direct and Factual:** Provide the factual answer directly. Do not start with conversational phrases like "According to the policy...".
3.  **Strictly Adhere to Context:** Do NOT use any information outside of the provided "DOCUMENT SNIPPETS". Do not make assumptions.
4.  **Handle Missing Information:** If the information required to answer the question is completely absent from the provided snippets, you must respond with the exact phrase: "Based on the provided document sections, the information is not available."

---
**DOCUMENT SNIPPETS:**

{context}
---

**QUESTION:**
{query}
---

**ANSWER:**
"""
            print("=== Gemini Prompt ===")
            print(prompt)
            print("=====================")

            model = genai.GenerativeModel(model_name=settings.LLM_MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=500
                )
            )

            try:
                answer = "".join(part.text for part in response.parts)
            except (ValueError, IndexError):
                logger.warning(f"Gemini response for query '{query[:50]}...' was blocked or empty.")
                answer = "The AI model's response was blocked or empty."
            
            return answer.strip() if answer else "The model did not generate a response for this query."
            
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            return "An error occurred while communicating with the language model."