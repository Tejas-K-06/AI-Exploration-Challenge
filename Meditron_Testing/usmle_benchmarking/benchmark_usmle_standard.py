import requests
import re
import time
import json
from datasets import load_dataset

# ==========================================
#      ⚙️ STANDARD USMLE CONFIGURATION
# ==========================================
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Number of questions to test
NUM_QUESTIONS = 25

# Temperature: 0.0 is best for "Standard" benchmarking (most likely answer)
TEMPERATURE = 0.6 

# Timeout
TIMEOUT_SECONDS = 180
# ==========================================

# --- SYSTEM PROMPT (STANDARD) ---
# No confidence instructions, just "Solve the problem"
SYSTEM_PROMPT = """You are an expert physician taking the USMLE Step 2 Clinical Knowledge exam.
1. Read the patient vignette carefully.
2. Answer based ONLY on clinical guidelines.

Format your response exactly like this:
Reasoning: [Step-by-step clinical reasoning]
Answer: [Option Letter A/B/C/D]"""

# --- ONE-SHOT EXAMPLE (No Confidence) ---
ONE_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: A 32-year-old woman comes to the office because of a 3-month history of fatigue... [Clinical details] ...\n"
            "Options:\n(A) Intravenous iron dextran\n(B) Oral ferrous sulfate\n(C) Erythropoietin injection\n(D) RBC transfusion\n\n"
            "Answer with the correct Option Letter."
        )
    },
    {
        "role": "assistant",
        "content": "Reasoning: The patient has iron deficiency anemia (Microcytic anemia with low Ferritin). The first-line treatment for stable iron deficiency anemia is oral iron supplementation.\nAnswer: B"
    }
]

def format_usmle_options(item):
    options_dict = item['options']
    formatted = []
    for letter in sorted(options_dict.keys()):
        formatted.append(f"({letter}) {options_dict[letter]}")
    return "\n".join(formatted)

def query_ollama_standard(question, options_text):
    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Answer with the correct Option Letter."
    )
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + ONE_SHOT_MESSAGES + [{"role": "user", "content": user_prompt}]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": 4096}
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        return {"content": data['message']['content'], "latency": time.time() - start_time}
    except Exception as e:
        return {"content": f"Error: {str(e)}", "latency": 0}

def extract_answer_standard(text):
    if not text: return "INVALID"
    
    # Simple extraction: Look for "Answer: X"
    match = re.search(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # Fallback: Look for the last capital letter A-D in the text
    clean_text = text.strip().upper()
    matches = re.findall(r'\b([A-D])\b', clean_text)
    if matches: return matches[-1]
    
    return "INVALID"

def main():
    print(f"\n🇺🇸 STARTING STANDARD USMLE BENCHMARK (No Safety Gate)")
    print(f"Questions: {NUM_QUESTIONS} | Temp: {TEMPERATURE}")
    print("="*60)

    try:
        data = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", streaming=True)
        data_iter = iter(data)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    results = []
    correct = 0
    dist = {"A": 0, "B": 0, "C": 0, "D": 0, "INVALID": 0}
    
    for i in range(NUM_QUESTIONS):
        try:
            item = next(data_iter)
        except StopIteration:
            break

        question = item['question']
        options_text = format_usmle_options(item)
        truth_letter = item['answer_idx']
        
        print(f"\n[Q{i+1}/{NUM_QUESTIONS}] Generating...")
        print(f"Question: {question[:100]}...") 
        
        res = query_ollama_standard(question, options_text)
        pred = extract_answer_standard(res['content'])
        
        status_icon = "✅" if pred == truth_letter else "❌"
        print(f"Model Output:\n{res['content'].strip()}")
        print(f"👉 Parsed: {pred} | Truth: {truth_letter} | {status_icon}")
        print("-" * 60)
        
        if pred in dist: dist[pred] += 1
        else: dist["INVALID"] += 1
        
        if pred == truth_letter: correct += 1
        
        results.append({
            "id": i+1,
            "question": question,
            "truth": truth_letter,
            "prediction": pred,
            "correct": (pred == truth_letter),
            "full_response": res['content']
        })

    acc = (correct / NUM_QUESTIONS) * 100
    fname = f"usmle_standard_T{str(TEMPERATURE).replace('.','')}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n{'='*30}")
    print(f"🇺🇸 FINAL RESULTS")
    print(f"Accuracy:      {acc:.2f}%")
    print(f"Distribution:  {dist}")
    print(f"Saved Log:     {fname}")
    print("="*30)

if __name__ == "__main__":
    main()