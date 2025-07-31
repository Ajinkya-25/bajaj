# quick_postgres_check.py - Simple PostgreSQL Connection Test
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def quick_test():
    """Quick PostgreSQL connection test"""
    print("🔍 Quick PostgreSQL Connection Test")
    print("-" * 40)

    # Check environment variables
    required_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB']
    for var in required_vars:
        value = os.getenv(var)
        if value:
            display = "*" * 8 if var == 'POSTGRES_PASSWORD' else value
            print(f"✓ {var}: {display}")
        else:
            print(f"✗ {var}: MISSING")
            return False

    print("\n🔌 Testing connection...")

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )

        # Test basic query
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, version();")
            db, user, version = cur.fetchone()

            print(f"✅ Connected successfully!")
            print(f"   Database: {db}")
            print(f"   User: {user}")
            print(f"   Version: {version.split(' ')[0]} {version.split(' ')[1]}")

        conn.close()
        print("\n🎉 PostgreSQL is working correctly!")
        return True

    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Common solutions:")
        print("1. Check your .env file has correct values")
        print("2. Verify host URL is correct")
        print("3. Check username/password")
        print("4. Ensure database exists")
        print("5. Check if your IP is whitelisted (for cloud DBs)")
        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    quick_test()