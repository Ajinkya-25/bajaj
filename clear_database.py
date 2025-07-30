
# clear_database.py
# This is a one-time utility script to clear all persistent data.
# Run this script once from your terminal to resolve the "stale data" issue.
# After running, restart the main application server.

import logging
from settings import settings
from postgres_manager import PostgresManager
from vectordb import FAISSManager

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_all_data():
    """
    Connects to Postgres and FAISS and clears all document-related data.
    """
    print("--- Starting Database Clearing Process ---")
    
    # --- Clear PostgreSQL Data ---
    try:
        logging.info("Attempting to clear PostgreSQL database...")
        pg_manager = PostgresManager()
        pg_manager.clear_all_data()
        logging.info("✅ PostgreSQL database cleared successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to clear PostgreSQL database: {e}", exc_info=True)
        print("---")
        print("⚠️ Could not clear PostgreSQL. Please check DB connection settings in your .env file.")
        print("---")


    # --- Clear FAISS Vector Index ---
    try:
        logging.info("Attempting to clear FAISS vector index...")
        # The dimension is needed for initialization but doesn't matter for clearing.
        faiss_manager = FAISSManager(dimension=settings.EMBEDDING_DIMENSION)
        faiss_manager.clear_index()
        logging.info("✅ FAISS index cleared successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to clear FAISS index: {e}", exc_info=True)
        print("---")
        print("⚠️ Could not clear the FAISS index. Check file permissions for the './data/' directory.")
        print("---")
        

    print("--- Database Clearing Process Finished ---")
    print("\nNext steps:")
    print("1. Stop the uvicorn server if it's running.")
    print("2. Restart the server with: uvicorn main:app --reload")
    print("3. Re-run your request. The system will now use the correctly processed data.")

if __name__ == "__main__":
    clear_all_data()