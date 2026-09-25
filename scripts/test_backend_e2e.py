"""
scripts/test_backend_e2e.py

End-to-end integration test of the FastAPI backend using a real PDF.
Tests the complete pipeline:
  PDF upload (POST /api/analyze)
  → Text extraction & cleaning
  → Rule-based section detection
  → Structured digest generation
  → TF-IDF summary generation
  → LED summary generation & ROUGE calculation (POST /api/rouge)
  → Session retrieval (GET /api/papers/{id})
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app

PDF_PATH = PROJECT_ROOT / "data" / "samples" / "attention.pdf"
HUMAN_ABSTRACT = (
    "The dominant sequence transduction models are based on complex recurrent or "
    "convolutional neural networks that include an encoder and a decoder. The best "
    "performing models also connect the encoder and decoder through an attention mechanism. "
    "We propose a new simple network architecture, the Transformer, based solely on attention "
    "mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two "
    "machine translation tasks show these models to be superior in quality while being more "
    "parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU "
    "on the WMT 2014 English-to-German translation task, improving over the existing best results, "
    "including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our "
    "model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for "
    "3.5 days on eight GPUs, a small fraction of the training costs of the best models from the "
    "literature. We show that the Transformer generalizes well to other tasks by applying it "
    "successfully to English constituency parsing both with large and limited training data."
)

def run_test():
    print("=" * 60)
    print("  FASTAPI BACKEND END-TO-END INTEGRATION TEST")
    print("=" * 60)
    print(f"Testing PDF: {PDF_PATH.name} ({PDF_PATH.stat().st_size:,} bytes)")

    client = TestClient(app)

    # -------------------------------------------------------------
    # 1. POST /api/analyze
    # -------------------------------------------------------------
    print("\n[Step 1] Uploading PDF to POST /api/analyze...")
    t0 = time.time()
    with open(PDF_PATH, "rb") as f:
        response = client.post(
            "/api/analyze",
            files=[("files", (PDF_PATH.name, f, "application/pdf"))]
        )
    upload_time = time.time() - t0

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "papers" in data, "Response missing 'papers' key"
    assert len(data["papers"]) == 1, f"Expected 1 paper, got {len(data['papers'])}"

    paper = data["papers"][0]
    paper_id = paper.get("paper_id")
    print(f"  Status code       : {response.status_code}")
    print(f"  Paper ID          : {paper_id}")
    print(f"  Filename          : {paper.get('filename')}")
    print(f"  Full text length  : {paper.get('full_text_length'):,} chars")
    print(f"  Analyze runtime   : {upload_time:.2f}s")

    # Verify extraction & cleaning
    assert paper.get("full_text_length", 0) > 0, "Extracted text is empty!"

    # Verify section detection
    metadata = paper.get("metadata", {})
    detected_sections = metadata.get("detected_sections", [])
    print(f"  Detected sections : {detected_sections}")
    assert len(detected_sections) > 0, "No sections detected!"

    # Verify structured digest
    digest = paper.get("digest", {})
    print("\n[Step 2] Structured Digest:")
    assert isinstance(digest, dict), "Digest is not a dictionary"
    expected_digest_keys = ["problem", "methodology", "dataset", "results", "limitations", "conclusion"]
    for k in expected_digest_keys:
        val = digest.get(k, "")
        preview = val[:80].replace("\n", " ") + "..." if len(val) > 80 else (val if val else "[Not detected]")
        print(f"  - {k:<12}: {preview}")
        assert k in digest, f"Missing digest key: {k}"

    # Verify TF-IDF summary
    tfidf_summary = paper.get("tfidf_summary", "")
    print(f"\n[Step 3] TF-IDF Summary:")
    print(f"  Length: {len(tfidf_summary)} chars")
    print(f"  Preview: {tfidf_summary[:120].replace(chr(10), ' ')}...")
    assert len(tfidf_summary.strip()) > 0, "TF-IDF summary is empty!"

    # -------------------------------------------------------------
    # 2. POST /api/rouge (triggers on-demand LED summary + ROUGE evaluation)
    # -------------------------------------------------------------
    print("\n[Step 4] Requesting POST /api/rouge (computes LED summary & ROUGE)...")
    t0 = time.time()
    rouge_payload = {
        "paper_id": paper_id,
        "reference_abstract": HUMAN_ABSTRACT
    }
    rouge_response = client.post("/api/rouge", json=rouge_payload)
    rouge_time = time.time() - t0

    assert rouge_response.status_code == 200, f"Expected 200, got {rouge_response.status_code}: {rouge_response.text}"
    rouge_data = rouge_response.json()
    print(f"  Status code       : {rouge_response.status_code}")
    print(f"  ROUGE runtime     : {rouge_time:.2f}s")

    # Verify ROUGE scores
    assert "tfidf" in rouge_data, "Missing tfidf in rouge response"
    assert "led" in rouge_data, "Missing led in rouge response"

    tf_scores = rouge_data["tfidf"]
    led_scores = rouge_data["led"]

    print("\n[Step 5] ROUGE Evaluation Results (vs Human Abstract):")
    print(f"  TF-IDF ROUGE: R1={tf_scores.get('rouge1')} | R2={tf_scores.get('rouge2')} | RL={tf_scores.get('rougeL')}")
    print(f"  LED    ROUGE: R1={led_scores.get('rouge1')} | R2={led_scores.get('rouge2')} | RL={led_scores.get('rougeL')}")

    for k in ["rouge1", "rouge2", "rougeL"]:
        assert k in tf_scores, f"Missing {k} in TF-IDF scores"
        assert k in led_scores, f"Missing {k} in LED scores"
        assert tf_scores[k] == tf_scores[k], f"NaN in TF-IDF {k}"
        assert led_scores[k] == led_scores[k], f"NaN in LED {k}"

    # -------------------------------------------------------------
    # 3. GET /api/papers/{paper_id} (verify full persisted session schema)
    # -------------------------------------------------------------
    print("\n[Step 6] Retrieving full updated session via GET /api/papers/{id}...")
    get_res = client.get(f"/api/papers/{paper_id}")
    assert get_res.status_code == 200, f"Expected 200, got {get_res.status_code}"
    session = get_res.json()

    print("  Verifying schema fields:")
    expected_fields = [
        "paper_id", "filename", "full_text_length", "led_summary",
        "tfidf_summary", "rouge", "digest", "metadata"
    ]
    for field in expected_fields:
        present = field in session
        print(f"    - {field:<18}: {'PRESENT' if present else 'MISSING'}")
        assert present, f"Missing field in session: {field}"

    # Verify LED summary is now populated
    led_summary = session.get("led_summary", "")
    print(f"\n[Step 7] LED Summary:")
    print(f"  Length: {len(led_summary)} chars")
    print(f"  Preview: {led_summary[:150].replace(chr(10), ' ')}...")
    assert len(led_summary.strip()) > 0, "LED summary is empty!"
    assert "nan" not in led_summary.lower(), "NaN in LED summary!"

    print("\n" + "=" * 60)
    print("  INTEGRATION TEST RESULT: ALL 9 CHECKS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
