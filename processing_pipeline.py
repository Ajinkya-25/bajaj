# pipeline/streamlined_processing_pipeline.py
import logging
import requests
import time
from typing import Dict, Any, List
from urllib.parse import urlparse
import os

from document_processor import DocumentProcessorFactory
from text_processor import EnhancedTextProcessor
from embedding_manager import EmbeddingManager
from vectordb import EnhancedFAISSManager
from postgres_manager import EnhancedPostgresManager

# Configure module-level logger
logger = logging.getLogger(__name__)


class StreamlinedProcessingPipeline:
    def __init__(self):
        self.text_processor = EnhancedTextProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = EnhancedFAISSManager(self.embedding_manager.get_dimension())
        self.postgres_db = EnhancedPostgresManager()
        logger.info("Initialized StreamlinedProcessingPipeline")

    def process_document_from_url(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """Process document from URL and answer questions in single pipeline"""
        start_time = time.time()
        logger.info(f"Starting streamlined processing for URL: {document_url}")

        try:
            # Step 1: Clear existing data for fresh document
            logger.info("Clearing existing database for new document...")
            self.postgres_db.clear_all_data()
            self.vector_db.clear_database()

            # Step 2: Download document
            logger.info("Downloading document from URL...")
            file_content, filename = self._download_document(document_url)

            # Step 3: Extract text
            processor = DocumentProcessorFactory.get_processor(filename)
            logger.info(f"Using processor: {processor.__class__.__name__}")
            raw_text = processor.extract_text(file_content)

            if not raw_text.strip():
                return {"error": "No text content extracted from document"}

            logger.info(f"Extracted {len(raw_text)} characters of text")

            # Step 4: Enhanced text processing with categorization
            logger.info("Processing text with category-based partitioning...")
            processed_text = self.text_processor.preprocess_text(raw_text)
            document_sections = self.text_processor.categorize_and_partition_document(
                processed_text,
                metadata={"document_url": document_url, "filename": filename}
            )

            if not document_sections:
                return {"error": "No chunks generated from document"}

            logger.info(f"Generated {len(document_sections)} categorized chunks")

            # Step 5: Extract categories for database storage
            categories = list(set(section['category'] for section in document_sections))
            logger.info(f"Document categories: {categories}")

            # Step 6: Store document metadata
            processing_time = time.time() - start_time
            document_id = self.postgres_db.store_document_with_url(
                filename=filename,
                document_url=document_url,
                file_type=processor.get_file_type(),
                file_size=len(file_content),
                categories=categories,
                processing_time=processing_time,
                metadata={
                    "original_length": len(raw_text),
                    "processed_length": len(processed_text),
                    "total_sections": len(document_sections)
                }
            )

            # Step 7: Generate embeddings for all chunks
            logger.info("Generating embeddings...")
            chunk_texts = [section['content'] for section in document_sections]
            embeddings = self.embedding_manager.generate_embeddings(chunk_texts)

            # Step 8: Prepare metadata for vector storage
            vector_metadata = []
            for i, section in enumerate(document_sections):
                metadata = section['metadata'].copy()
                metadata.update({
                    'document_id': document_id,
                    'chunk_index': i,
                    'category': section['category'],
                    'importance_score': section['importance_score'],
                    'filename': filename,
                    'document_url': document_url
                })
                vector_metadata.append(metadata)

            # Step 9: Store in vector database with categories
            logger.info("Storing embeddings in vector database...")
            vector_ids = self.vector_db.add_vectors_with_categories(embeddings, vector_metadata)

            # Step 10: Link vector IDs to chunks and store in PostgreSQL
            for i, section in enumerate(document_sections):
                section['vector_id'] = vector_ids[i]

            self.postgres_db.store_enhanced_chunks(document_id, document_sections)
            self.postgres_db.mark_document_processed(document_id)

            # Step 11: Answer all questions using the processed document
            logger.info(f"Answering {len(questions)} questions...")
            answers = self._answer_questions_efficiently(questions, document_id, raw_text, document_sections)

            total_time = time.time() - start_time
            logger.info(f"Completed streamlined processing in {total_time:.2f} seconds")

            return {
                "document_id": document_id,
                "document_url": document_url,
                "filename": filename,
                "categories": categories,
                "chunks_created": len(document_sections),
                "processing_time_seconds": total_time,
                "questions_answered": len(questions),
                "answers": answers,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Streamlined processing failed: {e}", exc_info=True)
            return {"error": f"Processing failed: {str(e)}"}

    def _download_document(self, url: str) -> tuple[bytes, str]:
        """Download document from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)

            # If no filename in URL, create one based on content type
            if not filename or '.' not in filename:
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    filename = 'document.pdf'
                elif 'word' in content_type or 'docx' in content_type:
                    filename = 'document.docx'
                else:
                    filename = 'document.txt'

            logger.info(f"Downloaded {filename}: {len(response.content)} bytes")
            return response.content, filename

        except Exception as e:
            logger.error(f"Failed to download document from {url}: {e}")
            raise

    def _answer_questions_efficiently(self, questions: List[str], document_id: int,
                                      full_text: str, document_sections: List[Dict]) -> List[Dict[str, Any]]:
        """Answer all questions efficiently using document context"""
        from query_pipeline import QueryPipeline
        import google.generativeai as genai
        from settings import settings

        answers = []

        try:
            # Prepare comprehensive context from all sections
            context_by_category = {}
            for section in document_sections:
                category = section['category']
                if category not in context_by_category:
                    context_by_category[category] = []
                context_by_category[category].append({
                    'content': section['content'],
                    'importance': section['importance_score']
                })

            # Sort sections by importance within each category
            for category in context_by_category:
                context_by_category[category].sort(key=lambda x: x['importance'], reverse=True)

            # Create comprehensive context (limit to avoid token limits)
            comprehensive_context = self._create_comprehensive_context(context_by_category)

            # Process all questions in a single LLM call for efficiency
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name=settings.LLM_MODEL)

            # Create multi-question prompt
            questions_text = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])

            prompt = f"""Based on the following document content, please answer all the questions accurately and concisely. 
Provide numbered answers corresponding to each question.

Document Content:
{comprehensive_context}

Questions:
{questions_text}

Please provide comprehensive answers based only on the provided document content. 
If any question cannot be answered from the content, state that clearly.
Format your response as:
1. [Answer to question 1]
2. [Answer to question 2]
etc."""

            start_time = time.time()
            response = model.generate_content(prompt)
            response_time = int((time.time() - start_time) * 1000)

            if response.text:
                # Parse the response to extract individual answers
                response_lines = response.text.strip().split('\n')
                current_answer = ""
                answer_index = 0

                for line in response_lines:
                    # Check if line starts with a number (new answer)
                    if line.strip() and line.strip()[0].isdigit() and '. ' in line:
                        # Save previous answer if exists
                        if current_answer and answer_index < len(questions):
                            answers.append({
                                "question": questions[answer_index],
                                "answer": current_answer.strip(),
                                "response_time_ms": response_time // len(questions),
                                "categories_used": list(context_by_category.keys()),
                                "source_document": f"document_id_{document_id}"
                            })
                            answer_index += 1

                        # Start new answer
                        current_answer = line.split('. ', 1)[1] if '. ' in line else line
                    else:
                        # Continue current answer
                        current_answer += " " + line.strip() if current_answer else line.strip()

                # Add the last answer
                if current_answer and answer_index < len(questions):
                    answers.append({
                        "question": questions[answer_index],
                        "answer": current_answer.strip(),
                        "response_time_ms": response_time // len(questions),
                        "categories_used": list(context_by_category.keys()),
                        "source_document": f"document_id_{document_id}"
                    })

                # Fill in any missing answers
                while len(answers) < len(questions):
                    answers.append({
                        "question": questions[len(answers)],
                        "answer": "Could not extract answer from the document content.",
                        "response_time_ms": response_time // len(questions),
                        "categories_used": [],
                        "source_document": f"document_id_{document_id}"
                    })

                # Store query results for analytics
                for answer in answers:
                    self.postgres_db.store_query_result_with_analytics(
                        query=answer["question"],
                        document_id=document_id,
                        categories_used=answer["categories_used"],
                        results=[{"answer": answer["answer"]}],
                        response_time_ms=answer["response_time_ms"]
                    )

            else:
                # Fallback: create error responses for all questions
                for question in questions:
                    answers.append({
                        "question": question,
                        "answer": "No response generated from the document.",
                        "response_time_ms": response_time,
                        "categories_used": [],
                        "source_document": f"document_id_{document_id}"
                    })

        except Exception as e:
            logger.error(f"Error answering questions: {e}", exc_info=True)
            # Create error responses for all questions
            for question in questions:
                answers.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "response_time_ms": 0,
                    "categories_used": [],
                    "source_document": f"document_id_{document_id}"
                })

        return answers

    def _create_comprehensive_context(self, context_by_category: Dict[str, List[Dict]],
                                      max_tokens: int = 15000) -> str:
        """Create comprehensive context from categorized sections"""
        context_parts = []
        current_tokens = 0

        # Prioritize categories by importance
        category_priorities = {
            'abstract': 10, 'conclusion': 9, 'results': 8, 'introduction': 7,
            'methodology': 6, 'discussion': 5, 'technical': 4, 'data': 3,
            'references': 1, 'general': 2
        }

        sorted_categories = sorted(
            context_by_category.keys(),
            key=lambda x: category_priorities.get(x, 2),
            reverse=True
        )

        for category in sorted_categories:
            if current_tokens >= max_tokens:
                break

            category_content = f"\n=== {category.upper()} ===\n"
            context_parts.append(category_content)
            current_tokens += len(category_content) // 4  # Rough token estimate

            for section in context_by_category[category]:
                if current_tokens >= max_tokens:
                    break

                content = section['content']
                content_tokens = len(content) // 4

                if current_tokens + content_tokens <= max_tokens:
                    context_parts.append(content + "\n")
                    current_tokens += content_tokens
                else:
                    # Add partial content to fit within limit
                    remaining_tokens = max_tokens - current_tokens
                    partial_content = content[:remaining_tokens * 4]
                    context_parts.append(partial_content + "...\n")
                    break

        return "".join(context_parts)