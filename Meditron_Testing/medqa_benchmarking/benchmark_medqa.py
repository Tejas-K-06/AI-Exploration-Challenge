import requests
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# --- CHAIN OF THOUGHT EXAMPLES ---
COT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: A 65-year-old man presents with sudden severe tearing chest pain radiating to the back. BP is 180/110 right arm, 130/70 left arm. Best diagnostic step?\n"
            "Options:\n(A) Echo\n(B) CT Angiography\n(C) MRI\n(D) Coronary Angio\n"
            "Answer the question using the following format:\n"
            "Reasoning: [Step-by-step logic]\n"
            "Answer: [Option Letter]"
        )
    },
    {
        "role": "assistant",
        "content": (
            "Reasoning: The patient describes 'tearing' chest pain radiating to the back, which is classic for aortic dissection. "
            "The blood pressure discrepancy between arms further supports this. "
            "Coronary angiography is for heart attacks (MI). MRI is too slow for an emergency. "
            "CT Angiography is the gold standard for diagnosing aortic dissection in a stable patient.\n"
            "Answer: B"
        )
    }
]

def extract_answer_cot(text):
    if not text: return None
    match = re.search(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches: return matches[-1]
    return None

def query_ollama_cot(question, options_dict):
    formatted_options = (
        f"(A) {options_dict['A']}\n"
        f"(B) {options_dict['B']}\n"
        f"(C) {options_dict['C']}\n"
        f"(D) {options_dict['D']}"
    )
    
    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n"
        f"Answer the question using the following format:\n"
        f"Reasoning: [Step-by-step logic]\n"
        f"Answer: [Option Letter]"
    )
    
    messages = COT_MESSAGES + [{"role": "user", "content": user_prompt}]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.6, 
            "num_ctx": 4096
        }
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        content = data['message']['content']
        
        tokens = data.get('eval_count', 0)
        eval_duration_ns = data.get('eval_duration', 1)
        tps = tokens / (eval_duration_ns / 1_000_000_000)
        
        return {
            "content": content, 
            "latency": end_time - start_time,
            "tps": tps
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"content": "", "latency": 0, "tps": 0}

def main():
    print(f"Loading MedQA (USMLE)...")
    try:
        data = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception as e:
        print(f"Error: {e}")
        return

    num_questions = 50 
    print(f"Starting CoT Benchmark on {num_questions} questions...")
    data = data.select(range(num_questions))

    correct = 0
    total = 0
    total_latency = 0
    results_log = []
    dist_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'INVALID': 0}

    print("="*80)
    
    # --- START TIMER HERE ---
    benchmark_start_time = time.time()
    
    for i, item in enumerate(data):
        question = item['question']
        ground_truth = item['answer_idx']
        options = item['options']
        
        res = query_ollama_cot(question, options)
        pred = extract_answer_cot(res['content'])

        if pred in dist_counts: dist_counts[pred] += 1
        else: dist_counts['INVALID'] += 1

        is_correct = (pred == ground_truth)
        if is_correct: correct += 1
        
        total += 1
        total_latency += res['latency']
        
        entry = {
            "id": i+1, "question": question, "truth": ground_truth, 
            "pred": pred, "correct": is_correct, 
            "latency_seconds": res['latency'],
            "tokens_per_second": res['tps'],
            "full_response": res['content']
        }
        results_log.append(entry)

        status = "✅" if is_correct else "❌"
        
        print(f"Q{i+1} | {status} | Time: {res['latency']:.2f}s | Speed: {res['tps']:.2f} t/s")
        print(f"Truth: {ground_truth} | Pred: {pred}")
        print("-" * 20 + " MODEL REASONING " + "-" * 20)
        print(res['content'].strip())
        print("=" * 80)

    # --- END TIMER HERE ---
    total_benchmark_time = time.time() - benchmark_start_time

    json_filename = 'medqa_cot_analytics.json'
    with open(json_filename, 'w') as f:
        json.dump(results_log, f, indent=4)

    acc = (correct / total) * 100
    avg_latency = total_latency / total
    
    print("\n" + "="*80)
    print(f"MEDQA (CoT) REPORT")
    print(f"Accuracy:    {acc:.2f}% ({correct}/{total})")
    print(f"Total Time:  {total_benchmark_time:.2f}s")  # <--- Added Total Time
    print(f"Avg Time:    {avg_latency:.2f}s")
    print(f"Distribution:{dist_counts}")
    print(f"Log saved to: {json_filename}")
    print("="*80)

if __name__ == "__main__":
    main()