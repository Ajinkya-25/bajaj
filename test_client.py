import requests
import json
import os
from urllib.parse import urlparse


class RAGAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def test_health(self):
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def download_and_upload_document(self, document_url: str):
        """Download document from URL and upload via API"""
        print(f"\n📥 Downloading and uploading document from: {document_url}")

        try:
            # Download document
            print("Downloading document...")
            doc_response = requests.get(document_url, timeout=30)
            doc_response.raise_for_status()

            # Get filename
            filename = os.path.basename(urlparse(document_url).path)
            if not filename or '.' not in filename:
                filename = 'document.pdf'

            print(f"Downloaded: {filename} ({len(doc_response.content)} bytes)")

            # Upload to API
            print("Uploading to RAG API...")
            files = {
                'file': (filename, doc_response.content, 'application/pdf')
            }

            upload_response = requests.post(
                f"{self.base_url}/upload-document",
                files=files
            )

            print(f"Upload Status: {upload_response.status_code}")

            if upload_response.status_code == 200:
                result = upload_response.json()
                print(f"✅ Upload successful!")
                print(f"   Document ID: {result.get('document_id', 'N/A')}")
                print(f"   Chunks created: {result.get('chunks_created', 'N/A')}")
                return result
            else:
                print(f"❌ Upload failed: {upload_response.text}")
                return None

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_single_query(self, query: str, top_k: int = 5):
        """Test single query endpoint"""
        print(f"\n🔍 Testing single query: {query}")

        try:
            payload = {
                "query": query,
                "top_k": top_k
            }

            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            print(f"Query Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Query successful!")
                print(f"   Answer: {result.get('answer', 'No answer')[:200]}...")
                print(f"   Sources found: {len(result.get('sources', []))}")
                return result
            else:
                print(f"❌ Query failed: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_multiple_queries(self, queries: list, top_k: int = 5):
        """Test multiple queries endpoint"""
        print(f"\n🔍 Testing multiple queries ({len(queries)} questions)")

        try:
            payload = {
                "queries": queries,
                "top_k": top_k
            }

            response = requests.post(
                f"{self.base_url}/multiple-queries",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            print(f"Multiple Queries Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Multiple queries successful!")

                for i, (query, query_result) in enumerate(zip(queries, result.get('results', []))):
                    print(f"\n   Query {i + 1}: {query}")
                    if 'error' not in query_result:
                        print(f"   Answer: {query_result.get('answer', 'No answer')[:150]}...")
                        print(f"   Sources: {len(query_result.get('sources', []))}")
                    else:
                        print(f"   ❌ Error: {query_result.get('error', 'Unknown error')}")

                return result
            else:
                print(f"❌ Multiple queries failed: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def run_complete_test(self, document_url: str, questions: list):
        """Run complete test workflow"""
        print("🚀 Starting Complete RAG API Test")
        print("=" * 60)

        # Step 1: Health check
        if not self.test_health():
            print("❌ Health check failed. Stopping test.")
            return

        # Step 2: Upload document
        upload_result = self.download_and_upload_document(document_url)
        if not upload_result:
            print("❌ Document upload failed. Stopping test.")
            return

        # Step 3: Test single query
        self.test_single_query(questions[0] if questions else "What is this document about?")

        # Step 4: Test multiple queries
        if len(questions) > 1:
            self.test_multiple_queries(questions)

        print("\n✅ Complete test finished!")


def main():
    """Main test function"""
    tester = RAGAPITester()

    # Test document URL (you can change this)
    document_url = "https://arxiv.org/pdf/2301.00001.pdf"

    # Test questions (you can modify these)
    questions = [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
        "What methodology or approach was used?",
        "What are the limitations mentioned?",
        "What future work is suggested?"
    ]

    # Run complete test
    tester.run_complete_test(document_url, questions)


def quick_test():
    """Quick test with a simple document"""
    tester = RAGAPITester()

    # Simple PDF for testing
    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    test_questions = [
        "What is this document about?",
        "Summarize the main content"
    ]

    tester.run_complete_test(test_url, test_questions)


def test_individual_endpoints():
    """Test endpoints individually"""
    tester = RAGAPITester()

    print("Testing individual endpoints...")

    # Just health check
    tester.test_health()

    # Just upload (you need to upload first before querying)
    # tester.download_and_upload_document("https://example.com/sample.pdf")

    # Just query (only works if you have uploaded a document)
    # tester.test_single_query("What is this document about?")


if __name__ == "__main__":
    print("RAG API Tester")
    print("Choose test type:")
    print("1. Complete test (upload + queries)")
    print("2. Quick test with simple PDF")
    print("3. Test individual endpoints")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    elif choice == "3":
        test_individual_endpoints()
    else:
        print("Running default complete test...")
        main()