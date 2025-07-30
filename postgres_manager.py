# database/enhanced_postgres_manager.py
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
        logger.info("Initializing EnhancedPostgresManager with params: %s:%s/%s",
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
        """Initialize or migrate database tables with enhanced schema"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create documents table with enhanced fields
                    cur.execute("DROP TABLE IF EXISTS query_results;")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            filename VARCHAR(255),
                            document_url TEXT,
                            file_type VARCHAR(50) NOT NULL,
                            file_size INTEGER,
                            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            processed BOOLEAN DEFAULT FALSE,
                            metadata JSONB,
                            categories JSONB,
                            total_chunks INTEGER DEFAULT 0,
                            processing_time_seconds FLOAT
                        )
                    """)

                    # Add new columns if they don't exist
                    new_columns = [
                        ('document_url', 'TEXT'),
                        ('categories', 'JSONB'),
                        ('total_chunks', 'INTEGER DEFAULT 0'),
                        ('processing_time_seconds', 'FLOAT')
                    ]

                    for column_name, column_type in new_columns:
                        cur.execute(f"""
                            ALTER TABLE documents
                            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                        """)

                    # Create enhanced document_chunks table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS document_chunks (
                            id SERIAL PRIMARY KEY,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            chunk_index INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            category VARCHAR(100) NOT NULL DEFAULT 'general',
                            importance_score FLOAT DEFAULT 0.5,
                            metadata JSONB,
                            vector_id INTEGER,
                            token_count INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Add new columns to chunks table
                    chunk_new_columns = [
                        ('category', 'VARCHAR(100) NOT NULL DEFAULT \'general\''),
                        ('importance_score', 'FLOAT DEFAULT 0.5'),
                        ('token_count', 'INTEGER')
                    ]

                    for column_name, column_type in chunk_new_columns:
                        cur.execute(f"""
                            ALTER TABLE document_chunks
                            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
                        """)

                    # Create category performance tracking table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS category_performance (
                            id SERIAL PRIMARY KEY,
                            category VARCHAR(100) NOT NULL,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            total_chunks INTEGER DEFAULT 0,
                            avg_importance FLOAT DEFAULT 0.0,
                            query_hit_count INTEGER DEFAULT 0,
                            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(category, document_id)
                        )
                    """)

                    # Create query results table with enhanced tracking
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS query_results (
                            id SERIAL PRIMARY KEY,
                            query TEXT NOT NULL,
                            document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                            categories_used JSONB,
                            results JSONB,
                            response_time_ms INTEGER,
                            accuracy_score FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Add indexes for better performance
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_chunks_category ON document_chunks(category)",
                        "CREATE INDEX IF NOT EXISTS idx_chunks_importance ON document_chunks(importance_score DESC)",
                        "CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON document_chunks(vector_id)",
                        "CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(processed)",
                        "CREATE INDEX IF NOT EXISTS idx_query_results_document ON query_results(document_id)"
                    ]

                    for index_sql in indexes:
                        cur.execute(index_sql)

                    conn.commit()
            logger.info("Enhanced database tables initialized/migrated successfully")
        except Exception as e:
            logger.error("Error initializing/migrating tables: %s", e, exc_info=True)
            raise

    def clear_all_data(self):
        """Clear all existing data to prepare for new document"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Delete in order to respect foreign key constraints
                    cur.execute("DELETE FROM query_results")
                    cur.execute("DELETE FROM category_performance")
                    cur.execute("DELETE FROM document_chunks")
                    cur.execute("DELETE FROM documents")

                    # Reset sequences
                    cur.execute("ALTER SEQUENCE documents_id_seq RESTART WITH 1")
                    cur.execute("ALTER SEQUENCE document_chunks_id_seq RESTART WITH 1")
                    cur.execute("ALTER SEQUENCE category_performance_id_seq RESTART WITH 1")
                    cur.execute("ALTER SEQUENCE query_results_id_seq RESTART WITH 1")

                    conn.commit()
            logger.info("Cleared all existing data from database")
        except Exception as e:
            logger.error("Error clearing database: %s", e, exc_info=True)
            raise

    def store_document_with_url(self, filename: str, document_url: str, file_type: str,
                                file_size: int, categories: List[str], processing_time: float,
                                metadata: Dict[str, Any] = None) -> int:
        """Store document metadata with URL and categories"""
        try:
            clean_metadata = self._convert_numpy_types(metadata or {})
            clean_categories = self._convert_numpy_types(categories)

            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO documents (filename, document_url, file_type, file_size, 
                                             categories, processing_time_seconds, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (filename, document_url, file_type, int(file_size),
                          json.dumps(clean_categories), float(processing_time),
                          json.dumps(clean_metadata)))
                    document_id = cur.fetchone()[0]
                    conn.commit()
            logger.info("Stored document '%s' with id=%s, categories=%s", filename, document_id, categories)
            return document_id
        except Exception as e:
            logger.error("Failed to store document '%s': %s", filename, e, exc_info=True)
            raise

    def store_enhanced_chunks(self, document_id: int, chunks: List[Dict[str, Any]]):
        """Store document chunks with category and importance information"""
        try:
            category_stats = {}

            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for i, chunk in enumerate(chunks):
                        clean_chunk = self._convert_numpy_types(chunk)
                        category = clean_chunk.get('category', 'general')
                        importance = clean_chunk.get('importance_score', 0.5)
                        token_count = clean_chunk.get('metadata', {}).get('token_count', 0)

                        cur.execute("""
                            INSERT INTO document_chunks (document_id, chunk_index, content, 
                                                       category, importance_score, metadata, 
                                                       vector_id, token_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            int(document_id),
                            int(i),
                            str(clean_chunk['content']),
                            str(category),
                            float(importance),
                            json.dumps(clean_chunk.get('metadata', {})),
                            int(clean_chunk.get('vector_id')) if clean_chunk.get('vector_id') is not None else None,
                            int(token_count)
                        ))

                        # Track category statistics
                        if category not in category_stats:
                            category_stats[category] = {'count': 0, 'total_importance': 0.0}
                        category_stats[category]['count'] += 1
                        category_stats[category]['total_importance'] += importance

                    # Update total chunks count
                    cur.execute("""
                        UPDATE documents SET total_chunks = %s WHERE id = %s
                    """, (len(chunks), int(document_id)))

                    # Store category performance data
                    for category, stats in category_stats.items():
                        avg_importance = stats['total_importance'] / stats['count']
                        cur.execute("""
                            INSERT INTO category_performance (category, document_id, total_chunks, avg_importance)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (category, document_id) 
                            DO UPDATE SET 
                                total_chunks = EXCLUDED.total_chunks,
                                avg_importance = EXCLUDED.avg_importance,
                                last_accessed = CURRENT_TIMESTAMP
                        """, (str(category), int(document_id), stats['count'], float(avg_importance)))

                    conn.commit()
            logger.info("Stored %d chunks for document id=%s with %d categories",
                        len(chunks), document_id, len(category_stats))
        except Exception as e:
            logger.error("Failed to store chunks for document id=%s: %s", document_id, e, exc_info=True)
            raise

    def get_chunks_by_vector_ids_with_categories(self, vector_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve chunks by vector IDs with category information"""
        try:
            clean_vector_ids = [int(vid) for vid in vector_ids]

            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT dc.*, d.filename, d.file_type, d.document_url
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.vector_id = ANY(%s)
                        ORDER BY dc.importance_score DESC, dc.vector_id
                    """, (clean_vector_ids,))
                    rows = [dict(row) for row in cur.fetchall()]
            logger.info("Retrieved %d chunks for vector IDs %s", len(rows), clean_vector_ids)
            return rows
        except Exception as e:
            logger.error("Failed to retrieve chunks for vector IDs %s: %s", vector_ids, e, exc_info=True)
            raise

    def get_chunks_by_categories(self, categories: List[str], limit: int = None,
                                 min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """Get chunks filtered by categories and importance"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT dc.*, d.filename, d.file_type, d.document_url
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.category = ANY(%s) AND dc.importance_score >= %s
                        ORDER BY dc.importance_score DESC, dc.category
                    """
                    params = [categories, float(min_importance)]

                    if limit:
                        query += " LIMIT %s"
                        params.append(int(limit))

                    cur.execute(query, params)
                    rows = [dict(row) for row in cur.fetchall()]
            logger.info("Retrieved %d chunks for categories %s", len(rows), categories)
            return rows
        except Exception as e:
            logger.error("Failed to retrieve chunks for categories %s: %s", categories, e, exc_info=True)
            raise

    def store_query_result_with_analytics(self, query: str, document_id: int,
                                          categories_used: List[str], results: List[Dict[str, Any]],
                                          response_time_ms: int, accuracy_score: float = None):
        """Store query results with analytics data"""
        try:
            clean_results = self._convert_numpy_types(results)
            clean_categories = self._convert_numpy_types(categories_used)

            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO query_results (query, document_id, categories_used, 
                                                 results, response_time_ms, accuracy_score)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (str(query), int(document_id), json.dumps(clean_categories),
                          json.dumps(clean_results), int(response_time_ms),
                          float(accuracy_score) if accuracy_score else None))

                    # Update category hit counts
                    for category in categories_used:
                        cur.execute("""
                            UPDATE category_performance 
                            SET query_hit_count = query_hit_count + 1,
                                last_accessed = CURRENT_TIMESTAMP
                            WHERE category = %s AND document_id = %s
                        """, (str(category), int(document_id)))

                    conn.commit()
            logger.info("Stored query result for query '%s' with categories %s", query, categories_used)
        except Exception as e:
            logger.error("Failed to store query result for '%s': %s", query, e, exc_info=True)
            raise

    def get_category_analytics(self, document_id: int = None) -> Dict[str, Any]:
        """Get analytics about category performance"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    if document_id:
                        cur.execute("""
                            SELECT * FROM category_performance 
                            WHERE document_id = %s
                            ORDER BY query_hit_count DESC, avg_importance DESC
                        """, (int(document_id),))
                    else:
                        cur.execute("""
                            SELECT category, 
                                   SUM(total_chunks) as total_chunks,
                                   AVG(avg_importance) as avg_importance,
                                   SUM(query_hit_count) as total_hits,
                                   MAX(last_accessed) as last_accessed
                            FROM category_performance 
                            GROUP BY category
                            ORDER BY total_hits DESC, avg_importance DESC
                        """)

                    rows = [dict(row) for row in cur.fetchall()]
            return {'categories': rows}
        except Exception as e:
            logger.error("Failed to get category analytics: %s", e, exc_info=True)
            return {'categories': []}

    def get_current_document_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current document"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM documents 
                        ORDER BY upload_time DESC 
                        LIMIT 1
                    """)
                    row = cur.fetchone()
                    return dict(row) if row else None
        except Exception as e:
            logger.error("Failed to get current document info: %s", e, exc_info=True)
            return None

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