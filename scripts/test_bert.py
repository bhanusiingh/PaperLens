import sys
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf.extractor import PDFExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.summarization.led_summarizer import LEDSummarizer

def test_bert():
    pdf_path = "data/samples/1810.04805.pdf"
    
    print("Extracting text from BERT...")
    raw = PDFExtractor(pdf_path).extract_text()
    clean = TextCleaner().clean(raw)
    
    led = LEDSummarizer()
    tokenizer, _ = led._load_model()
    
    # Test our warning fix logic
    print("Tokenizing to check length...")
    t_ids = tokenizer.encode(clean, add_special_tokens=False)
    print(f"Total tokens in BERT paper: {len(t_ids)}")
    
    print("Running summarization (should trigger chunking silently)...")
    t0 = time.time()
    summary = led.summarize(clean)
    t1 = time.time()
    
    print(f"Summarization complete in {t1-t0:.2f} seconds.")
    print(f"Output Length: {len(summary)} characters.")
    print("Summary snippet:")
    print(summary[:500] + "...")

if __name__ == '__main__':
    test_bert()
