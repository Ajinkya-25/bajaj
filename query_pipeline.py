# pipeline/query_pipeline.py
from typing import List, Dict, Any
from embedding_manager import EmbeddingManager
from vectordb import FAISSManager
from postgres_manager import PostgresManager
import google.generativeai as genai
from settings import settings


class QueryPipeline:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_db = FAISSManager(self.embedding_manager.get_dimension())
        self.postgres_db = PostgresManager()
        genai.configure(api_key=settings.GEMINI_API_KEY)

    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a single query through the pipeline"""
        try:
            # Step 1: Generate Query Embedding
            query_embedding = self.embedding_manager.generate_query_embedding(query)

            # Step 2: Vector Similarity Search
            scores, vector_ids = self.vector_db.search(query_embedding, k=top_k)

            if not vector_ids or vector_ids[0] == -1:
                return {"error": "No relevant documents found"}

            # Step 3: Retrieve Matching Documents
            valid_vector_ids = [vid for vid in vector_ids if vid != -1]
            chunks = self.postgres_db.get_chunks_by_vector_ids(valid_vector_ids)

            if not chunks:
                return {"error": "No matching document chunks found"}

            # Step 4: Context Preparation
            context = self._prepare_context(query, chunks, scores[:len(chunks)])

            # Step 5: Generate Answer using Gemini
            answer = self._generate_answer(query, context)

            # Step 6: Store results
            results = {
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "filename": chunk["filename"],
                        "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                        "score": float(score)
                    }
                    for chunk, score in zip(chunks, scores[:len(chunks)])
                ]
            }

            self.postgres_db.store_query_result(query, results["sources"])

            return results

        except Exception as e:
            return {"error": f"Query processing failed: {str(e)}"}

    def process_multiple_queries(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Process multiple queries"""
        results = []
        for query in queries:
            result = self.process_query(query, top_k)
            results.append(result)
        return results

    def _prepare_context(self, query: str, chunks: List[Dict], scores: List[float]) -> str:
        """Prepare context from retrieved chunks"""
        context_parts = []
        for chunk, score in zip(chunks, scores):
            context_parts.append(
                f"Source: {chunk['filename']}\n"
                f"Content: {chunk['content']}\n"
                f"Relevance Score: {score:.3f}\n"
            )

        return "\n---\n".join(context_parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini API with optimized prompt"""
        try:
            # Optimized prompt for token efficiency
            prompt = f"""Context: {context[:2000]}

Question: {query}

Answer based only on the context above. Be specific and include relevant details like numbers, dates, conditions. If information is not in context, state clearly."""

            model = genai.GenerativeModel(model_name=settings.LLM_MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=300,  # Limit response length
                    top_p=0.8
                )
            )

            return response.text.strip() if response.text else "No response generated"

        except Exception as e:
            return f"Error generating answer: {str(e)}"