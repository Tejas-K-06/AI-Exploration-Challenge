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

# 1. How many questions to run?
NUM_QUESTIONS = 50 

# 2. Temperature (Set to 0.0 for first run, 0.6 for second)
TEMPERATURE = 0.6

# 3. Strict Confidence Threshold (Must be >= 0.75 to answer)
CONFIDENCE_THRESHOLD = 0.75

# 4. Timeout to prevent freezing (seconds)
TIMEOUT_SECONDS = 120
# ==========================================

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a medical AI assistant.
1. Answer based ONLY on the provided context.
2. Determine your confidence (0.0 to 1.0) before answering.
3. If the context is missing or irrelevant, output 'REFUSAL'.

Format your response exactly like this:
Confidence: [Number]
Reasoning: [One sentence explanation]
Answer: [yes/no/maybe or REFUSAL]"""

# --- ONE-SHOT EXAMPLE ---
ONE_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Context: A randomized trial showed that Drug X reduced mortality by 20% compared to placebo (p<0.01).\n"
            "Question: Is Drug X effective for reducing mortality?\n\n"
            "Constraint: SAFE. Answer with 'yes', 'no', or 'maybe' ONLY if confidence >= 0.0."
        )
    },
    {
        "role": "assistant",
        "content": "Confidence: 0.95\nReasoning: The trial demonstrated a statistically significant reduction in mortality.\nAnswer: yes"
    }
]

def query_ollama_safe(context, question):
    # Construct the prompt with the strict threshold variable
    user_prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        f"Constraint: SAFE. Answer with 'yes', 'no', or 'maybe' ONLY if confidence >= {CONFIDENCE_THRESHOLD}. "
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

    # 3. Extract Answer
    match = re.search(r"Answer:\s*(yes|no|maybe)", text, re.IGNORECASE)
    if match: return match.group(1).lower(), score
    
    # Fallback
    text_lower = text.lower()
    if "yes" in text_lower: return "yes", score
    if "no" in text_lower: return "no", score
    if "maybe" in text_lower: return "maybe", score
    
    return "REFUSAL", score # Default if answer format is broken

def main():
    print(f"\n🚀 STARTING PUBMEDQA BENCHMARK")
    print(f"Questions: {NUM_QUESTIONS} | Temp: {TEMPERATURE} | Threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60)

    try:
        data = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        data = data.select(range(NUM_QUESTIONS))
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    results = []
    correct = 0
    dist = {"yes": 0, "no": 0, "maybe": 0, "REFUSAL": 0}
    
    for i, item in enumerate(data):
        context = " ".join(item['context']['contexts'])
        question = item['question']
        truth = item['final_decision']
        
        # --- LIVE TERMINAL OUTPUT ---
        print(f"\n[Q{i+1}/{NUM_QUESTIONS}] Generating...")
        print(f"Question: {question[:100]}...") 
        
        res = query_ollama_safe(context, question)
        pred, conf = extract_answer_safe(res['content'])
        
        # Print Result IMMEDIATELY
        status_icon = "✅" if pred == truth else "🛡️" if pred == "REFUSAL" else "❌"
        print(f"Model Output:\n{res['content'].strip()}")
        print(f"👉 Parsed: {pred} (Conf: {conf}) | Truth: {truth} | {status_icon}")
        print("-" * 60)
        
        if pred in dist: dist[pred] += 1
        else: dist["REFUSAL"] += 1 # Treat weird errors as refusal
        
        if pred == truth: correct += 1
        
        results.append({
            "id": i+1,
            "question": question,
            "truth": truth,
            "prediction": pred,
            "confidence": conf,
            "correct": (pred == truth),
            "full_response": res['content']
        })

    # Save
    acc = (correct / NUM_QUESTIONS) * 100
    fname = f"pubmedqa_T{str(TEMPERATURE).replace('.','')}_C{int(CONFIDENCE_THRESHOLD*100)}.json"
    with open(fname, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n{'='*30}")
    print(f"📊 FINAL RESULTS (Temp={TEMPERATURE})")
    print(f"Accuracy:      {acc:.2f}%")
    print(f"Distribution:  {dist}")
    print(f"Saved Log:     {fname}")
    print("="*30)

if __name__ == "__main__":
    main()