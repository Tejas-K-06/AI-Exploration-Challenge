import requests
import re
import time
import json
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
#        ⚙️ BENCHMARK CONFIGURATION
# ==========================================
MODEL_NAME = "meditron:7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# 1. Number of questions to test
NUM_QUESTIONS = 50 

# 2. Temperature (Start with 0.0, then try 0.6)
TEMPERATURE = 0.6 

# 3. Strict Confidence Threshold
CONFIDENCE_THRESHOLD = 0.75

# 4. Timeout (Prevents freezing)
TIMEOUT_SECONDS = 120
# ==========================================

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a medical AI assistant taking a clinical exam.
1. Answer based ONLY on your internal medical knowledge.
2. Determine your confidence (0.0 to 1.0) before answering.
3. If you are unsure, output 'REFUSAL'.

Format your response exactly like this:
Confidence: [Number]
Reasoning: [One sentence explanation]
Answer: [Option Letter A/B/C/D or REFUSAL]"""

# --- ONE-SHOT EXAMPLE (MCQ Format) ---
ONE_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: A 24-year-old female presents with anemia. Peripheral smear shows spherocytes. Which test confirms the diagnosis?\n"
            "Options:\n(A) Coombs test\n(B) Osmotic fragility test\n(C) Hb electrophoresis\n(D) Bone marrow aspiration\n\n"
            "Constraint: SAFE. Answer with Option Letter ONLY if confidence >= 0.0."
        )
    },
    {
        "role": "assistant",
        "content": "Confidence: 0.95\nReasoning: Spherocytes on peripheral smear suggest Hereditary Spherocytosis, which is confirmed by the Osmotic Fragility Test.\nAnswer: B"
    }
]

def format_options(item):
    """Helper to format A/B/C/D options from the dataset"""
    options = [item['opa'], item['opb'], item['opc'], item['opd']]
    formatted = []
    for i, opt in enumerate(options):
        formatted.append(f"({chr(65+i)}) {opt}")
    return "\n".join(formatted)

def query_ollama_safe(question, options_text):
    user_prompt = (
        f"Question: {question}\n"
        f"Options:\n{options_text}\n\n"
        f"Constraint: SAFE. Answer with Option Letter ONLY if confidence >= {CONFIDENCE_THRESHOLD}. "
        f"Otherwise say REFUSAL."
    )
    
    # Construct history: System -> One-Shot -> User
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
    
    # 1. Parse Confidence
    score = 0.0
    conf_match = re.search(r"Confidence:\s*([0-1]?\.\d+)", text)
    if conf_match:
        try:
            score = float(conf_match.group(1))
        except ValueError: pass

    # 2. Strict Threshold Check
    if score < CONFIDENCE_THRESHOLD:
        return "REFUSAL", score

    # 3. Extract Option Letter (A, B, C, D)
    match = re.search(r"Answer:\s*\(?([A-D])\)?", text, re.IGNORECASE)
    if match: return match.group(1).upper(), score
    
    # Fallback search
    clean_text = text.strip().upper()
    matches = re.findall(r'\b([A-D])\b', clean_text)
    if matches: return matches[-1], score
    
    return "REFUSAL", score

def main():
    print(f"\n🚀 STARTING MEDMCQA BENCHMARK")
    print(f"Questions: {NUM_QUESTIONS} | Temp: {TEMPERATURE} | Threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60)

    try:
        # --- FIXED LINE: streaming=True ---
        data = load_dataset("medmcqa", split="validation", streaming=True)
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
        options_text = format_options(item)
        
        # MedMCQA answers are indices 0=A, 1=B, 2=C, 3=D
        truth_idx = item['cop'] 
        truth_letter = chr(65 + truth_idx)
        
        # --- LIVE TERMINAL OUTPUT ---
        print(f"\n[Q{i+1}/{NUM_QUESTIONS}] Generating...")
        print(f"Question: {question[:100]}...") 
        
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
    fname = f"medmcqa_T{str(TEMPERATURE).replace('.','')}_C{int(CONFIDENCE_THRESHOLD*100)}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n{'='*30}")
    print(f"📊 MEDMCQA RESULTS (Temp={TEMPERATURE})")
    print(f"Accuracy:      {acc:.2f}%")
    print(f"Distribution:  {dist}")
    print(f"Saved Log:     {fname}")
    print("="*30)

if __name__ == "__main__":
    main()