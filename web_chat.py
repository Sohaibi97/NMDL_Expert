# web_chat.py
# srun -p gpu --gres=gpu:a100_40gb:1 --cpus-per-task=4 --mem=64G --export=ALL,PORT=8000 python web_chat.py
import os
import socket
from flask import Flask, request, jsonify, send_from_directory

import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
from peft import PeftModel

# --- Offline erzwingen (HPC ohne Internet) ---
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

# === Pfade ===
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/khamlichi/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88",
)
LORA_PATH = os.environ.get(
    "LORA_MODEL_PATH",
    "/home/khamlichi/Projekt_NMDL_2/outputs/ministral3b-instruct-nmdl-lora",
)

SYSTEM_MESSAGE = (
    "Du bist ein Experte für NMDL. Antworte kurz, präzise und ohne Wiederholungen."
    "Wenn eine Information nicht eindeutig bekannt oder nicht öffentlich dokumentiert ist, "
    "sage ausdrücklich, dass sie unbekannt oder projektspezifisch ist, und erfinde nichts. "
    "Beantworte Fragen zu NMDL gemäß der im Training vermittelten Definition und Struktur. "
    "Beantworte Begrüßungen und Smalltalk natürlich und kurz, ohne Fachbegriffe zu erzwingen. "
    "Wenn eine Frage unklar ist, bitte um Präzisierung statt zu raten. "
    "Wenn du eine Liste von Begriffen nennst, beende sie mit einem vollständigen Satz und wiederhole keine Elemente."
)

print("🚀 Lade Ministral-3B Instruct + LoRA ...")

# Tokenizer aus LoRA-Ordner laden (damit Template/Tokenizer konsistent ist)
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Base-Model laden
model = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    device_map={"": 0} if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
    trust_remote_code=True,
)

# LoRA Adapter laden
model = PeftModel.from_pretrained(model, LORA_PATH, local_files_only=True)
model.eval()

if not torch.cuda.is_available():
    model.to("cpu")

print("💬 Ministral-3B Instruct Web-Server ist bereit.")

# === Gesprächsverlauf (letzte N Turns) ===
history_turns = []
MAX_RECENT_TURNS = 6


def generate_answer(user_text: str) -> str:
    global history_turns

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE}
    ]

    for t in history_turns[-MAX_RECENT_TURNS:]:
        messages.append({"role": "user", "content": t["user"]})
        messages.append({"role": "assistant", "content": t["assistant"]})

    messages.append({"role": "user", "content": user_text})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # NUR neu generierte Tokens dekodieren (ohne Prompt/History)
    generated_ids = out[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not answer:
        answer = "(keine Antwort generiert)"

    history_turns.append({"user": user_text, "assistant": answer})
    if len(history_turns) > MAX_RECENT_TURNS:
        history_turns = history_turns[-MAX_RECENT_TURNS:]

    return answer


# === Flask-App ===
app = Flask(__name__)


@app.route("/")
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, "chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()
    if not user_text:
        return jsonify({"answer": ""}), 400

    answer = generate_answer(user_text)
    return jsonify({"answer": answer})


@app.route("/reset", methods=["POST"])
def reset_conversation():
    global history_turns
    history_turns = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    node = (
        os.environ.get("SLURMD_NODENAME")
        or os.environ.get("SLURM_NODELIST")
        or os.environ.get("SLURM_JOB_NODELIST")
    )

    if not node:
        node = socket.gethostname()

    print("\n🔐 SSH Tunnel:")
    print(f"ssh -N -L 18001:{node}:{port} khamlichi@login1.hpc.uni-potsdam.de")
    print("🌐 Browser:")
    print("http://localhost:18001/\n")

    app.run(host="0.0.0.0", port=port, debug=False)