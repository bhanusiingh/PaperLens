import sys
import time
import os
import psutil
import urllib.request
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

def get_device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"
    except Exception:
        return "CPU (Torch not loaded)"

def download_arxiv_paper(arxiv_id, out_path):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    resp = urllib.request.urlopen(url)
    xml_data = resp.read()
    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    entry = root.find('atom:entry', ns)
    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
    
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    if not os.path.exists(out_path):
        urllib.request.urlretrieve(pdf_url, out_path)
    
    return abstract

def run_task1():
    print("=== TASK 1: CACHED RUN ===")
    pdf_path = "data/samples/attention.pdf"
    
    process = psutil.Process(os.getpid())
    
    t0 = time.time()
    raw = PDFExtractor(pdf_path).extract_text()
    clean = TextCleaner().clean(raw)
    
    t_load_start = time.time()
    led = LEDSummarizer()
    led._load_model()
    t_load_end = time.time()
    
    t_inf_start = time.time()
    summary = led.summarize(clean)
    t_inf_end = time.time()
    
    t_total_end = time.time()
    
    # Get peak memory (peak_wset is available on Windows)
    mem_info = process.memory_info()
    peak_ram = getattr(mem_info, 'peak_wset', mem_info.rss) / (1024**2)
    
    print(f"1. Model loading time: {t_load_end - t_load_start:.2f} seconds")
    print(f"2. Actual inference time: {t_inf_end - t_inf_start:.2f} seconds")
    print(f"3. Total runtime: {t_total_end - t0:.2f} seconds")
    print(f"4. Device used: {get_device_info()}")
    print(f"5. Peak RAM usage: {peak_ram:.2f} MB\n")

def run_task2():
    print("=== TASK 2: 5 DIFFERENT ARXIV PAPERS ===")
    # Using famous moderate-length NLP papers
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
    run_task1()
    run_task2()
