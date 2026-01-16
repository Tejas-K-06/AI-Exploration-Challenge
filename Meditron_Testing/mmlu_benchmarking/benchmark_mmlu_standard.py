import requests
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
TEMPERATURE = 0.6 # CoT needs creativity

# --- MEDICAL SUBSETS OF MMLU ---
# We combine these to create a "Medical MMLU" score
SUBSETS = [
    'clinical_knowledge',
    'medical_genetics',
    'anatomy',
    'professional_medicine',
    'college_biology'
]

# --- CoT FEW-SHOT EXAMPLE (4 Options) ---
COT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: What is the primary mechanism of action of ibuprofen?\n"
            "Options:\n"
            "(A) Stimulation of mu-opioid receptors\n"
            "(B) Inhibition of cyclooxygenase (COX) enzymes\n"
            "(C) Blockade of sodium channels\n"
            "(D) Antagonism of H1 receptors\n"
            "Answer the question using the following format:\n"
            "Reasoning: [Step-by-step logic]\n"
            "Answer: [Option Letter]"
        )
    },
    {
        "role": "assistant",
        "content": (
            "Reasoning: Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID). "
            "NSAIDs work by inhibiting the cyclooxygenase (COX) enzymes, which converts arachidonic acid to prostaglandins. "
            "Opioids stimulate mu-receptors (A). Local anesthetics block sodium channels (C). Antihistamines block H1 (D). "
            "Therefore, the mechanism is COX inhibition.\n"
            "Answer: B"
        )
    }
]

def extract_answer(text):
    if not text: return None
    match = re.search(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    matches = re.findall(r'\b([A-D])\b', text.upper())
    if matches: return matches[-1]
    return None

def query_ollama(question, options):
    formatted_opts = (
        f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}\n(D) {options[3]}"
    )
    
    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n{formatted_opts}\n"
        f"Answer the question using the following format:\n"
        f"Reasoning: [Step-by-step logic]\n"
        f"Answer: [Option Letter]"
    )
    
    messages = COT_MESSAGES + [{"role": "user", "content": user_prompt}]
    
    payload = {
        "model": MODEL_NAME, "messages": messages, "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": 4096}
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return {"content": data['message']['content'], "latency": time.time() - start_time}
    except Exception as e:
        print(f"Error: {e}")
        return {"content": "", "latency": 0}

def main():
    print(f"Loading Standard MMLU (Medical Subsets)...")
    all_data = []
    
    # Load 10 questions from each medical subset
    for sub in SUBSETS:
        try:
            print(f"  - Loading {sub}...")
            # 'cais/mmlu' is the official repo
            ds = load_dataset("cais/mmlu", sub, split="test")
            # Select 10 from each to make a 50-question exam
            subset_data = ds.select(range(10))
            for item in subset_data:
                item['subset'] = sub # Tag the source
                all_data.append(item)
        except Exception as e:
            print(f"    Failed to load {sub}: {e}")

    print(f"\nBenchmarking {len(all_data)} questions across 5 medical domains...")
    print("="*80)

    correct = 0
    total = 0
    results_log = []
    benchmark_start = time.time()

    for i, item in enumerate(all_data):
        question = item['question']
        options = item['choices']
        ground_truth_idx = item['answer'] # 0,1,2,3
        ground_truth = ['A', 'B', 'C', 'D'][ground_truth_idx]
        
        res = query_ollama(question, options)
        pred = extract_answer(res['content'])
        
        is_correct = (pred == ground_truth)
        if is_correct: correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"Q{i+1} [{item['subset']}] | {status} | Truth: {ground_truth} | Pred: {pred}")
        
        results_log.append({
            "id": i+1, "subset": item['subset'], "question": question, 
            "truth": ground_truth, "pred": pred, "reasoning": res['content']
        })

    total_time = time.time() - benchmark_start
    acc = (correct / total) * 100
    
    # Save
    with open('mmlu_standard_analytics.json', 'w') as f:
        json.dump(results_log, f, indent=4)

    print("\n" + "="*80)
    print(f"STANDARD MMLU (MEDICAL) REPORT")
    print(f"Accuracy:    {acc:.2f}% ({correct}/{total})")
    print(f"Total Time:  {total_time:.2f}s")
    print("="*80)

if __name__ == "__main__":
    main()