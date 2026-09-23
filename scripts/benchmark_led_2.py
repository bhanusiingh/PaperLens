import sys
import time
import os
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pdf.extractor import PDFExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.summarization.led_summarizer import LEDSummarizer
from src.evaluation.rouge_evaluator import ROUGEEvaluator

def download_arxiv_paper(arxiv_id, out_path):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = requests.get(url, headers=headers)
    xml_data = resp.content
    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entry = root.find('atom:entry', ns)
    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
    
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if not os.path.exists(out_path):
        print(f"  Downloading PDF {arxiv_id}...")
        pdf_resp = requests.get(pdf_url, headers=headers)
        if pdf_resp.status_code == 200:
            with open(out_path, 'wb') as f:
                f.write(pdf_resp.content)
        else:
            raise Exception(f"Failed to download {pdf_url}: {pdf_resp.status_code}")
    
    return abstract

def run_task2():
    print("=== TASK 2: 5 DIFFERENT ARXIV PAPERS ===")
    papers = [
        "1810.04805", # BERT
        "1907.11692", # RoBERTa
        "1908.10084", # Sentence-BERT
        "1910.13461", # BART
        "1409.0473"   # Bahdanau Attention
    ]
    
    led = LEDSummarizer()
    rouge_eval = ROUGEEvaluator()
    results = []
    
    for pid in papers:
        print(f"Testing Paper ID: {pid}")
        out_pdf = f"data/samples/{pid}.pdf"
        abstract = download_arxiv_paper(pid, out_pdf)
        
        raw = PDFExtractor(out_pdf).extract_text()
        clean = TextCleaner().clean(raw)
        
        t_inf_start = time.time()
        summary = led.summarize(clean)
        inf_time = time.time() - t_inf_start
        
        scores = rouge_eval.score(summary, abstract).to_dict()
        
        results.append({
            "inf_time": inf_time,
            "length": len(summary),
            "r1": scores['rouge1'],
            "r2": scores['rouge2'],
            "rl": scores['rougeL']
        })
        
        print(f"  - Inference time: {inf_time:.2f} seconds")
        print(f"  - Output length: {len(summary)} characters")
        print(f"  - ROUGE-1: {scores['rouge1']:.4f}")
        print(f"  - ROUGE-2: {scores['rouge2']:.4f}")
        print(f"  - ROUGE-L: {scores['rougeL']:.4f}\n")
    
    mean_inf = sum(r['inf_time'] for r in results) / 5
    mean_len = sum(r['length'] for r in results) / 5
    mean_r1 = sum(r['r1'] for r in results) / 5
    mean_r2 = sum(r['r2'] for r in results) / 5
    mean_rl = sum(r['rl'] for r in results) / 5
    
    print("=== MEAN SCORES (5 Papers) ===")
    print(f"Mean Inference Time: {mean_inf:.2f} seconds")
    print(f"Mean Output Length: {mean_len:.1f} characters")
    print(f"Mean ROUGE-1: {mean_r1:.4f}")
    print(f"Mean ROUGE-2: {mean_r2:.4f}")
    print(f"Mean ROUGE-L: {mean_rl:.4f}")

if __name__ == '__main__':
    run_task2()
