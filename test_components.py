# test_components.py - Test each component individually
import traceback


def test_settings():
    """Test settings configuration"""
    try:
        print("Testing settings...")
        from settings import settings
        print(f"✓ Settings loaded")
        print(f"  - FAISS path: {settings.FAISS_INDEX_PATH}")
        print(f"  - Postgres host: {settings.POSTGRES_HOST}")
        print(f"  - Gemini API configured: {bool(settings.GEMINI_API_KEY)}")
        return True
    except Exception as e:
        print(f"✗ Settings failed: {e}")
        traceback.print_exc()
        return False


def test_postgres():
    """Test PostgreSQL connection"""
    try:
        print("\nTesting PostgreSQL connection...")
        from postgres_manager import PostgresManager
        pg = PostgresManager()
        print("✓ PostgreSQL manager initialized")
        return True
    except Exception as e:
        print(f"✗ PostgreSQL failed: {e}")
        traceback.print_exc()
        return False


def test_embedding():
    """Test embedding manager"""
    try:
        print("\nTesting Embedding Manager...")
        from embedding_manager import EmbeddingManager
        em = EmbeddingManager()
        print("✓ Embedding manager initialized")

        # Test a simple embedding
        test_text = "This is a test"
        embeddings = em.generate_embeddings(test_text)
        print(f"✓ Generated embedding: shape {embeddings.shape}")
        return True
    except Exception as e:
        print(f"✗ Embedding manager failed: {e}")
        traceback.print_exc()
        return False


def test_vectordb():
    """Test vector database"""
    try:
        print("\nTesting Vector Database...")
        from vectordb import FAISSManager
        vdb = FAISSManager(768)
        print(f"✓ FAISS manager initialized with {vdb.get_total_vectors()} vectors")
        return True
    except Exception as e:
        print(f"✗ Vector database failed: {e}")
        traceback.print_exc()
        return False


def test_document_processor():
    """Test document processor"""
    try:
        print("\nTesting Document Processor...")
        from document_processor import DocumentProcessorFactory
        processor = DocumentProcessorFactory.get_processor("test.txt")
        print(f"✓ Document processor: {processor.get_file_type()}")
        return True
    except Exception as e:
        print(f"✗ Document processor failed: {e}")
        traceback.print_exc()
        return False


def test_processing_pipeline():
    """Test processing pipeline"""
    try:
        print("\nTesting Processing Pipeline...")
        from processing_pipeline import ProcessingPipeline
        pp = ProcessingPipeline()
        print("✓ Processing pipeline initialized")
        return True
    except Exception as e:
        print(f"✗ Processing pipeline failed: {e}")
        traceback.print_exc()
        return False


def test_query_pipeline():
    """Test query pipeline"""
    try:
        print("\nTesting Query Pipeline...")
        from query_pipeline import QueryPipeline
        qp = QueryPipeline()
        print("✓ Query pipeline initialized")
        return True
    except Exception as e:
        print(f"✗ Query pipeline failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== RAG System Component Tests ===\n")

    tests = [
        test_settings,
        test_postgres,
        test_embedding,
        test_vectordb,
        test_document_processor,
        test_processing_pipeline,
        test_query_pipeline
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n=== Results: {passed}/{len(tests)} tests passed ===")

    if passed == len(tests):
        print("✓ All components working! Try the API now.")
    else:
        print("✗ Some components failed. Check the errors above.")