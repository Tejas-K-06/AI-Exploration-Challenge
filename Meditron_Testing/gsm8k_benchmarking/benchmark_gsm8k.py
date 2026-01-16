import requests
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# --- SIMULATED CONVERSATION HISTORY (Few-Shot) ---
FEW_SHOT_HISTORY = [
    {"role": "user", "content": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"},
    {"role": "assistant", "content": "There are 15 trees originally. Then there were 21 trees after some were planted. So there must have been 21 - 15 = 6 trees planted.\n#### 6"},
    {"role": "user", "content": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"},
    {"role": "assistant", "content": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5 cars.\n#### 5"},
    {"role": "user", "content": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"},
    {"role": "assistant", "content": "Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"}
]

# --- Helper Function: Extract Number ---
def extract_answer_number(text):
    if not text: return None
    
    # 1. Isolate the final answer after "####" if it exists
    if "####" in text:
        text = text.split("####")[-1]
    
    # 2. Remove commas (e.g., 24,000 -> 24000)
    text = text.replace(',', '')
    
    # 3. Find the LAST number 
    matches = re.findall(r'-?\d+\.?\d*', text)
    
    if matches:
        return matches[-1]
    return None

# --- Helper Function: Query Ollama ---
def query_ollama(current_question):
    messages = FEW_SHOT_HISTORY + [{"role": "user", "content": current_question}]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096}
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
            "tokens": tokens,
            "tps": tps
        }
    except Exception as e:
        print(f"Error querying model: {e}")
        return {"content": "", "latency": 0, "tokens": 0, "tps": 0}

def main():
    print(f"Loading GSM8K dataset...")
    try:
        data = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- CONTROL NUMBER OF QUESTIONS ---
    num_questions = 25  # Set this to 50 for your final run
    print(f"Selecting the first {num_questions} questions...")
    data = data.select(range(num_questions))

    correct = 0
    total = 0
    total_latency = 0
    total_tokens = 0
    results_log = [] 

    print(f"\nStarting benchmark on {MODEL_NAME}...\n")
    print("="*80)
    
    benchmark_start_time = time.time()

    for i, item in enumerate(data):
        question = item['question']
        raw_answer = item['answer']
        ground_truth_num = raw_answer.split('####')[-1].strip()
        
        # Query Model
        result_data = query_ollama(question)
        model_response = result_data['content']
        latency = result_data['latency']
        tps = result_data['tps']
        
        model_pred_num = extract_answer_number(model_response)

        # Compare
        is_correct = False
        if model_pred_num:
            clean_pred = str(float(model_pred_num)).rstrip('0').rstrip('.')
            clean_truth = str(float(ground_truth_num)).rstrip('0').rstrip('.')
            if clean_pred == clean_truth:
                is_correct = True
                correct += 1
        
        total += 1
        total_latency += latency
        total_tokens += result_data['tokens']
        
        # Build Dictionary for JSON
        entry = {
            "id": i + 1,
            "question": question,
            "ground_truth": ground_truth_num,
            "prediction": model_pred_num,
            "is_correct": is_correct,
            "latency_seconds": round(latency, 2),
            "tokens_per_second": round(tps, 2),
            "full_model_response": model_response
        }
        results_log.append(entry)

        # --- DETAILED LOGGING ---
        print(f"Question {i+1} | Time: {latency:.2f}s | Speed: {tps:.2f} t/s")
        print(f"Q: {question}") 
        print(f"\n[Model Answer]:\n{model_response.strip()}")
        print(f"\n[Expected]:  {ground_truth_num}")
        print(f"[Predicted]: {model_pred_num}  ({'✅ CORRECT' if is_correct else '❌ WRONG'})")
        print("-" * 80)

    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time

    # --- SAVE JSON ---
    # UPDATED FILENAME HERE
    json_filename = 'gsm8k_benchmark_analytics.json'
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=4)

    # --- FINAL ANALYTICS DASHBOARD ---
    avg_latency = total_latency / total if total > 0 else 0
    avg_tps = total_tokens / total_latency if total_latency > 0 else 0
    accuracy = (correct / total) * 100

    print(f"\n{'='*30} BENCHMARK REPORT {'='*30}")
    print(f"Model:                 {MODEL_NAME}")
    print(f"Total Questions:       {total}")
    print(f"Accuracy:              {accuracy:.2f}% ({correct}/{total})")
    print(f"Total Benchmark Time:  {total_benchmark_time:.2f} seconds")
    print(f"Avg Time Per Question: {avg_latency:.2f} seconds")
    print(f"Avg Generation Speed:  {avg_tps:.2f} tokens/second")
    print(f"Log saved to:          {json_filename}")
    print("="*80)

if __name__ == "__main__":
    main()