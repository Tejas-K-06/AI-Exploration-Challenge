import requests
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# --- FEW-SHOT EXAMPLES ---
FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Complete the description with the most plausible ending:\n"
            "Context: A woman is seen sitting in a chair with a pair of scissors. She holds the scissors up to her hair.\n"
            "Options:\n"
            "[0] She cuts her own hair.\n"
            "[1] She stands up and walks away.\n"
            "[2] She eats the scissors.\n"
            "[3] She paints the scissors red.\n"
            "Answer with the correct option number."
        )
    },
    {"role": "assistant", "content": "0"},
    
    {
        "role": "user",
        "content": (
            "Complete the description with the most plausible ending:\n"
            "Context: The man is playing a piano on stage. He finishes the song and looks at the audience.\n"
            "Options:\n"
            "[0] The audience leaves silently.\n"
            "[1] The man smashes the piano.\n"
            "[2] The audience applauds loudly.\n"
            "[3] The man starts flying.\n"
            "Answer with the correct option number."
        )
    },
    {"role": "assistant", "content": "2"}
]

# --- Helper Function: Extract Option Number ---
def extract_option_index(text):
    if not text: return None
    # Look for single digits 0, 1, 2, or 3
    matches = re.findall(r'[0-3]', text)
    if matches:
        return int(matches[-1]) 
    return None

# --- Helper Function: Query Ollama ---
def query_ollama(context, options):
    formatted_options = "\n".join([f"[{i}] {opt}" for i, opt in enumerate(options)])
    user_prompt = (
        f"Complete the description with the most plausible ending:\n"
        f"Context: {context}\n"
        f"Options:\n{formatted_options}\n"
        f"Answer with the correct option number."
    )
    
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": user_prompt}]
    
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
    print(f"Loading HellaSwag dataset...")
    try:
        data = load_dataset("hellaswag", split="validation")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- CONTROL NUMBER OF QUESTIONS ---
    num_questions = 25  # Set this to 50 for your final run
    print(f"Selecting the first {num_questions} questions from Validation split...")
    data = data.select(range(num_questions))

    correct = 0
    total = 0
    total_latency = 0
    total_tokens = 0
    results_log = []

    print(f"\nStarting HellaSwag benchmark on {MODEL_NAME}...\n")
    print("="*80)
    
    benchmark_start_time = time.time()

    for i, item in enumerate(data):
        ctx = item['ctx']
        endings = item['endings']
        ground_truth_idx = int(item['label']) 
        
        # Query Model
        result_data = query_ollama(ctx, endings)
        model_response = result_data['content']
        latency = result_data['latency']
        tps = result_data['tps']
        
        # Parse Prediction
        pred_idx = extract_option_index(model_response)

        # Check correctness
        is_correct = False
        if pred_idx is not None and pred_idx == ground_truth_idx:
            is_correct = True
            correct += 1
        
        total += 1
        total_latency += latency
        total_tokens += result_data['tokens']
        
        # Determine Status String
        if pred_idx is None:
            status_str = "INVALID (No 0-3 found)"
        elif pred_idx == ground_truth_idx:
            status_str = "✅ CORRECT"
        else:
            status_str = "❌ WRONG"

        # Log Entry
        entry = {
            "id": i + 1,
            "context": ctx,
            "options": endings,
            "ground_truth_index": ground_truth_idx,
            "ground_truth_text": endings[ground_truth_idx],
            "prediction_index": pred_idx,
            "prediction_text": endings[pred_idx] if (pred_idx is not None and pred_idx < 4) else "INVALID",
            "is_correct": is_correct,
            "latency_seconds": round(latency, 2),
            "tokens_per_second": round(tps, 2),
            "full_model_response": model_response
        }
        results_log.append(entry)

        # --- DETAILED LOGGING (Visual Update) ---
        print(f"Q{i+1} | Time: {latency:.2f}s | Speed: {tps:.2f} t/s")
        print(f"Context: {ctx}")
        print("\nOptions:")
        for idx, opt in enumerate(endings):
            # Highlight the Correct Answer with [CORRECT]
            # Highlight the Model's Choice with [PREDICTED]
            prefix = f"[{idx}] "
            suffix = ""
            if idx == ground_truth_idx:
                suffix += " <--- [TRUTH]"
            if idx == pred_idx:
                suffix += " <--- [MODEL CHOICE]"
            print(f"  {prefix}{opt}{suffix}")

        print(f"\n[Raw Model Output]: {model_response.strip().replace(chr(10), ' ')}")
        print(f"Result: {status_str}")
        print("-" * 80)

    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time

    # --- SAVE JSON ---
    json_filename = 'hellaswag_benchmark_analytics.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=4)

    # --- FINAL REPORT ---
    avg_latency = total_latency / total if total > 0 else 0
    avg_tps = total_tokens / total_latency if total_latency > 0 else 0
    accuracy = (correct / total) * 100

    print(f"\n{'='*30} HELLASWAG REPORT {'='*30}")
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