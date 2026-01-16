import requests
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.6
CATEGORY_FILTER = "health"  # Options: health, biology, business, chemistry, etc.

# --- 10-OPTION MAPPING ---
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# --- FEW-SHOT CoT EXAMPLE (Health Specific) ---
# We show the model how to handle 10 options
COT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: A patient presents with elevated localized pruritus. Which of the following is the most appropriate initial pharmacologic treatment?\n"
            "Options:\n"
            "(A) Oral prednisone\n(B) Topical hydrocortisone\n(C) IV diphenhydramine\n(D) Oral cephalexin\n"
            "(E) Topical mupirocin\n(F) Oral loratadine\n(G) Topical capsaicin\n(H) IV epinephrine\n"
            "(I) Oral ibuprofen\n(J) Topical lidocaine\n"
            "Answer the question using the following format:\n"
            "Reasoning: [Step-by-step logic]\n"
            "Answer: [Option Letter]"
        )
    },
    {
        "role": "assistant",
        "content": (
            "Reasoning: The patient has localized pruritus (itching). Systemic steroids (A) or IV antihistamines (C) are too aggressive for a localized issue. "
            "Antibiotics (D, E) treat infection, not itching. "
            "Topical hydrocortisone is a low-potency steroid specifically indicated for localized inflammation and pruritus. "
            "It is the standard first-line conservative therapy.\n"
            "Answer: B"
        )
    }
]

def format_options(options_list):
    text = ""
    for idx, opt_text in enumerate(options_list):
        if idx < len(LETTERS):
            text += f"({LETTERS[idx]}) {opt_text}\n"
    return text

def extract_answer(text):
    if not text: return None
    # Look for "Answer: X"
    match = re.search(r"Answer:\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # Fallback: Look for the last single letter A-J
    matches = re.findall(r'\b([A-J])\b', text.upper())
    if matches: return matches[-1]
    return None

def query_ollama(question, options_list):
    formatted_opts = format_options(options_list)
    
    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_opts}\n"
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
            "temperature": TEMPERATURE, 
            "num_ctx": 4096
        }
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        end_time = time.time()
        
        data = response.json()
        return {
            "content": data['message']['content'], 
            "latency": end_time - start_time,
            "tps": data.get('eval_count', 0) / (data.get('eval_duration', 1) / 1e9)
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"content": "", "latency": 0, "tps": 0}

def main():
    print(f"Loading MMLU-Pro (Category: {CATEGORY_FILTER})...")
    try:
        # Load the dataset from Hugging Face
        data = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        
        # Filter for category
        data = data.filter(lambda x: x['category'] == CATEGORY_FILTER)
        print(f"Found {len(data)} questions in '{CATEGORY_FILTER}'.")
        
    except Exception as e:
        print(f"Error: {e}")
        return

    num_questions = 10
    print(f"Benchmarking first {num_questions} questions...")
    data = data.select(range(min(num_questions, len(data))))

    correct = 0
    total = 0
    total_latency = 0
    results_log = []
    dist_counts = {l: 0 for l in LETTERS}
    dist_counts['INVALID'] = 0

    print("="*80)
    
    benchmark_start_time = time.time()
    
    for i, item in enumerate(data):
        question = item['question']
        options = item['options']
        # Answer index is 0-9 in dataset, map to Letter
        ground_truth_idx = item['answer_index']
        ground_truth = LETTERS[ground_truth_idx]
        
        res = query_ollama(question, options)
        pred = extract_answer(res['content'])

        if pred in dist_counts: dist_counts[pred] += 1
        else: dist_counts['INVALID'] += 1

        is_correct = (pred == ground_truth)
        if is_correct: correct += 1
        
        total += 1
        total_latency += res['latency']
        
        # Log entry
        status = "✅" if is_correct else "❌"
        print(f"Q{i+1} | {status} | Truth: {ground_truth} | Pred: {pred}")
        print(f"   💭 {res['content'].strip()[:150]}...") # Preview thought
        print("-" * 80)
        
        results_log.append({
            "id": i+1, "question": question, "truth": ground_truth, 
            "pred": pred, "correct": is_correct, "full_response": res['content']
        })

    total_time = time.time() - benchmark_start_time
    acc = (correct / total) * 100
    
    # Save
    with open('mmlu_pro_analytics.json', 'w') as f:
        json.dump(results_log, f, indent=4)

    print("\n" + "="*80)
    print(f"MMLU-PRO REPORT (Category: {CATEGORY_FILTER})")
    print(f"Accuracy:    {acc:.2f}% ({correct}/{total})")
    print(f"Total Time:  {total_time:.2f}s")
    print(f"Distribution:{dist_counts}")
    print("="*80)

if __name__ == "__main__":
    main()