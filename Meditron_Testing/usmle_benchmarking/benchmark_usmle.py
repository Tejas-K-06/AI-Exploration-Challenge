import requests
import re
import time
import json
from datasets import load_dataset

# ==========================================
#        ⚙️ USMLE BENCHMARK CONFIG
# ==========================================
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 1. Number of questions (USMLE is slow, start with 20)
NUM_QUESTIONS = 25

# 2. Temperature (Stick to 0.6 for best calibration)
TEMPERATURE = 0.6

# 3. Confidence Threshold
CONFIDENCE_THRESHOLD = 0.75

# 4. Timeout (USMLE vignettes are long, give it time)
TIMEOUT_SECONDS = 180
# ==========================================

# --- SYSTEM PROMPT (USMLE SPECIFIC) ---
SYSTEM_PROMPT = """You are an expert physician taking the USMLE Step 2 Clinical Knowledge exam.
1. Read the patient vignette carefully.
2. Answer based ONLY on clinical guidelines.
3. Determine your confidence (0.0 to 1.0).
4. If you are unsure, output 'REFUSAL'.

Format your response exactly like this:
Confidence: [Number]
Reasoning: [Step-by-step clinical reasoning]
Answer: [Option Letter A/B/C/D or REFUSAL]"""

# --- ONE-SHOT EXAMPLE (Clinical Vignette Style) ---
ONE_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: A 32-year-old woman comes to the office because of a 3-month history of fatigue and generalized weakness. She has no history of serious illness. Physical examination shows no abnormalities. Laboratory studies show:\n"
            "Hemoglobin: 9.2 g/dL\nMCV: 72 fL\nFerritin: 10 ng/mL\n"
            "Which of the following is the most appropriate next step in management?\n"
            "Options:\n(A) Intravenous iron dextran\n(B) Oral ferrous sulfate\n(C) Erythropoietin injection\n(D) RBC transfusion\n\n"
            "Constraint: SAFE. Answer with Option Letter ONLY if confidence >= 0.0."
        )
    },
    {
        "role": "assistant",
        "content": "Confidence: 0.95\nReasoning: The patient has iron deficiency anemia (Microcytic anemia with low Ferritin). The first-line treatment for stable iron deficiency anemia is oral iron supplementation.\nAnswer: B"
    }
]

def format_usmle_options(item):
    """
    Parses MedQA options which come as a dictionary {'A': 'text', 'B': 'text'...}
    """
    options_dict = item['options'] # keys are A, B, C, D
    formatted = []
    # Sort keys to ensure A, B, C, D order
    for letter in sorted(options_dict.keys()):
        formatted.append(f"({letter}) {options_dict[letter]}")
    return "\n".join(formatted)

def query_ollama_safe(question, options_text):
    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Constraint: SAFE. Answer with Option Letter ONLY if confidence >= {CONFIDENCE_THRESHOLD}. "
        f"Otherwise say REFUSAL."
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
        return {"content": f"REFUSAL (Error: {str(e)})", "latency": 0}

def extract_answer_safe(text):
    if not text: return "REFUSAL", 0.0
    
    score = 0.0
    conf_match = re.search(r"Confidence:\s*([0-1]?\.\d+)", text)
    if conf_match:
        try: score = float(conf_match.group(1))
        except: pass

    if score < CONFIDENCE_THRESHOLD:
        return "REFUSAL", score

    # Extract Answer (A/B/C/D)
    match = re.search(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match: return match.group(1).upper(), score
    
    clean_text = text.strip().upper()
    matches = re.findall(r'\b([A-D])\b', clean_text)
    if matches: return matches[-1], score
    
    return "REFUSAL", score

def main():
    print(f"\n🇺🇸 STARTING USMLE (MedQA) BENCHMARK")
    print(f"Questions: {NUM_QUESTIONS} | Temp: {TEMPERATURE} | Threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60)

    try:
        # GBaker/MedQA-USMLE-4-options is a clean version of the dataset
        print("Loading dataset (streaming)...")
        data = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", streaming=True)
        data_iter = iter(data)
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    results = []
    correct = 0
    dist = {"A": 0, "B": 0, "C": 0, "D": 0, "REFUSAL": 0}
    
    for i in range(NUM_QUESTIONS):
        try:
            item = next(data_iter)
        except StopIteration:
            break

        question = item['question']
        options_text = format_usmle_options(item)
        truth_letter = item['answer_idx'] # In this dataset, truth is 'A', 'B', etc.
        
        # --- LIVE TERMINAL OUTPUT ---
        print(f"\n[Q{i+1}/{NUM_QUESTIONS}] Generating...")
        print(f"Question: {question[:120]}...") 
        
        res = query_ollama_safe(question, options_text)
        pred, conf = extract_answer_safe(res['content'])
        
        status_icon = "✅" if pred == truth_letter else "🛡️" if pred == "REFUSAL" else "❌"
        print(f"Model Output:\n{res['content'].strip()}")
        print(f"👉 Parsed: {pred} (Conf: {conf}) | Truth: {truth_letter} | {status_icon}")
        print("-" * 60)
        
        if pred in dist: dist[pred] += 1
        else: dist["REFUSAL"] += 1
        
        if pred == truth_letter: correct += 1
        
        results.append({
            "id": i+1,
            "question": question,
            "truth": truth_letter,
            "prediction": pred,
            "confidence": conf,
            "correct": (pred == truth_letter),
            "full_response": res['content']
        })

    acc = (correct / NUM_QUESTIONS) * 100
    fname = f"usmle_T{str(TEMPERATURE).replace('.','')}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n{'='*30}")
    print(f"🇺🇸 USMLE RESULTS (Temp={TEMPERATURE})")
    print(f"Accuracy:      {acc:.2f}%")
    print(f"Distribution:  {dist}")
    print(f"Saved Log:     {fname}")
    print("="*30)

if __name__ == "__main__":
    main()