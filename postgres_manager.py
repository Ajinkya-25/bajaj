import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import json
import numpy as np
from datetime import datetime
from settings import settings

# Configure module-level logger
logger = logging.getLogger(__name__)


class PostgresManager:
    def __init__(self):
        self.connection_params = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'database': settings.POSTGRES_DB,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD
        }
        logger.info("Initializing PostgresManager with params: %s:%s/%s",
                    self.connection_params['host'],
                    self.connection_params['port'],
                    self.connection_params['database'])
        self.init_tables()

    def get_connection(self):
        """Get PostgreSQL connection with UTF-8 encoding"""
        conn = psycopg2.connect(
            **self.connection_params,
            client_encoding='utf8'
        )
        # Set connection to autocommit for better UTF-8 handling
        conn.set_client_encoding('UTF8')
        return conn

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        return obj

    def init_tables(self):
        """Initialize or migrate database tables"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create documents table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            filename VARCHAR(255),
                            file_type VARCHAR(50) NOT NULL,
                            file_size INTEGER,
                            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            processed BOOLEAN DEFAULT FALSE,
                            metadata JSONB
                        )
                    """)

                    # Ensure all columns exist (for migrations)
                    cur.execute("""
                        ALTER TABLE documents
                        ADD COLUMN IF NOT EXISTS filename VARCHAR(255)
                    """)

                    cur.execute("""
                        ALTER TABLE documents
                        ADD COLUMN IF NOT EXISTS file_type VARCHAR(50)
                    """)

                    # Create document_chunks table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id SERIAL PRIMARY KEY,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            chunk_index INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            metadata JSONB,
                            vector_id INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Create query_results table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS query_results (
                            id SERIAL PRIMARY KEY,
                            query TEXT NOT NULL,
                            results JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    conn.commit()
            logger.info("Database tables initialized/migrated successfully")
        except Exception as e:
            logger.error("Error initializing/migrating tables: %s", e, exc_info=True)
            raise

    def store_document(self, filename: str, file_type: str, file_size: int, metadata: Dict[str, Any] = None) -> int:
        """Store document metadata and return document ID"""
        try:
            # Convert numpy types in metadata
            clean_metadata = self._convert_numpy_types(metadata or {})

            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO documents (filename, file_type, file_size, metadata)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (filename, file_type, int(file_size), json.dumps(clean_metadata)))
                    document_id = cur.fetchone()[0]
                    conn.commit()
            logger.info("Stored document '%s' with id=%s", filename, document_id)
            return document_id
        except Exception as e:
            logger.error("Failed to store document '%s': %s", filename, e, exc_info=True)
            raise

    def store_chunks(self, document_id: int, chunks: List[Dict[str, Any]]):
        """Store document chunks"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for i, chunk in enumerate(chunks):
                        # Convert numpy types in chunk data
                        clean_chunk = self._convert_numpy_types(chunk)

                        cur.execute("""
                            INSERT INTO document_chunks (document_id, chunk_index, content, metadata, vector_id)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            int(document_id),
                            int(i),
                            str(clean_chunk['content']),
                            json.dumps(clean_chunk.get('metadata', {})),
                            int(clean_chunk.get('vector_id')) if clean_chunk.get('vector_id') is not None else None
                        ))
                    conn.commit()
            logger.info("Stored %d chunks for document id=%s", len(chunks), document_id)
        except Exception as e:
            logger.error("Failed to store chunks for document id=%s: %s", document_id, e, exc_info=True)
            raise

    def get_chunks_by_vector_ids(self, vector_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve chunks by vector IDs"""
        try:
            # Convert numpy.int64 to regular Python int
            clean_vector_ids = [int(vid) for vid in vector_ids]

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT dc.*, d.filename, d.file_type
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.vector_id = ANY(%s)
                        ORDER BY dc.vector_id
                    """, (clean_vector_ids,))
                    rows = [dict(row) for row in cur.fetchall()]
            logger.info("Retrieved %d chunks for vector IDs %s", len(rows), clean_vector_ids)
            return rows
        except Exception as e:
            logger.error("Failed to retrieve chunks for vector IDs %s: %s", vector_ids, e, exc_info=True)
            raise

    def mark_document_processed(self, document_id: int):
        """Mark document as processed"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE documents SET processed = TRUE WHERE id = %s",
                        (int(document_id),)
                    )
                    conn.commit()
            logger.info("Marked document id=%s as processed", document_id)
        except Exception as e:
            logger.error("Failed to mark document id=%s as processed: %s", document_id, e, exc_info=True)
            raise

    def store_query_result(self, query: str, results: List[Dict[str, Any]]):
        """Store query results"""
        try:
            # Convert numpy types in results
            clean_results = self._convert_numpy_types(results)

            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO query_results (query, results)
                        VALUES (%s, %s)
                    """, (str(query), json.dumps(clean_results)))
                    conn.commit()
            logger.info("Stored query result for query '%s'", query)
        except Exception as e:
            logger.error("Failed to store query result for '%s': %s", query, e, exc_info=True)
            raise