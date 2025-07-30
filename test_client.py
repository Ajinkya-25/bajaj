import requests
import json


class RAGAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def test_health(self):
        print("🔍 Testing /health endpoint (optional)...")
        try:
            response = requests.get(f"{self.base_url}/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"⚠️ Health check failed (endpoint may not exist): {e}")

    def process_document_with_questions(self, document_url: str, questions: list):
        print(f"\n🚀 Submitting document and questions to /process")
        try:
            payload = {
                "document_url": document_url,
                "questions": questions
            }

            response = requests.post(
                f"{self.base_url}/process",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            print(f"Response Status: {response.status_code}")
            if response.status_code == 200:
                answers = response.json().get("answers", [])
                for i, (q, a) in enumerate(zip(questions, answers)):
                    print(f"\nQ{i+1}: {q}\nA{i+1}: {a}")
            else:
                print(f"❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")


def run_test():
    tester = RAGAPITester()

    # Optional health check (only if you implement /health)
    tester.test_health()

    # Test document and questions
    document_url = "https://arxiv.org/pdf/2301.00001.pdf"
    questions = [
        "What is the main topic of this document?",
        "What are the key findings or conclusions?",
        "What methodology or approach was used?",
        "What are the limitations mentioned?",
        "What future work is suggested?"
    ]

    tester.process_document_with_questions(document_url, questions)


if __name__ == "__main__":
    run_test()
