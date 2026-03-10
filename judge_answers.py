#!/usr/bin/env python3
# judge_answers.py
#
# Erwartet Testdaten in:  Test_data/*.jsonl
# Jede Zeile: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
# => "assistant" wird als Erwartungshorizont (Referenz) genutzt.
#
# Das Script testet:
# 1) Base Model ohne Fine-Tuning
# 2) Fine-Tuned Model (Base + LoRA)
#
# Für jede Frage werden beide Modelle bewertet und getrennt geloggt.
#
# Outputs:
# - judge_outputs/<Taxonomie>_judge_base.jsonl
# - judge_outputs/<Taxonomie>_judge_ft.jsonl
# - judge_logs/<Taxonomie>_scores_base.jsonl
# - judge_logs/<Taxonomie>_scores_ft.jsonl
# - judge_outputs/summary.json

import os
import json
import glob
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Offline erzwingen (HPC ohne Internet) ---
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

# ===== Pfade =====
TEST_DIR = os.environ.get("TEST_DIR", "Test_data")

CANDIDATE_BASE_MODEL_PATH = os.environ.get(
    "CANDIDATE_BASE_MODEL_PATH",
    "/home/khamlichi/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88",
)
CANDIDATE_LORA_PATH = os.environ.get(
    "CANDIDATE_LORA_PATH",
    "/home/khamlichi/Projekt_NMDL_2/outputs/ministral3b-instruct-nmdl-lora",
)

JUDGE_MODEL_PATH = os.environ.get(
    "JUDGE_MODEL_PATH",
    "/home/khamlichi/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
)

OUT_DIR = os.environ.get("OUT_DIR", "judge_outputs")
LOG_DIR = os.environ.get("LOG_DIR", "judge_logs")
PLOT_DIR = os.environ.get("PLOT_DIR", "judge_plots")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
# ===== Candidate System Prompt =====
CANDIDATE_SYSTEM = os.environ.get(
    "CANDIDATE_SYSTEM",
    "Du bist ein Experte für NMDL. Beantworte präzise und fachlich korrekt."
)

# ===== Judge Prompt =====
JUDGE_PROMPT = """Du bist ein strenger, objektiver Bewertungsassistent.
Du vergleichst eine MODELL-ANTWORT mit einem ERWARTUNGSHORIZONT (Ground Truth).
Bewertet wird ausschließlich die MODELL-ANTWORT: sachliche Korrektheit und Abdeckung der Kerninhalte.
Stil, Länge, Höflichkeit sind irrelevant.

WICHTIG:
- Inhalt vor Wortlaut: Paraphrasen sind OK, solange der Sachverhalt korrekt ist.
- Fachbegriffe, Eigennamen und definierte Bezeichnungen müssen präzise korrekt sein.
- Korrekte Zusatzinformationen dürfen die Bewertung nicht verschlechtern.
- Bewerte streng, aber konsistent.
- Nutze nur die Klassen gut, mittel oder schlecht.
- Du MUSST GENAU EIN JSON-OBJEKT ausgeben, nichts anderes.
- Kein Markdown, keine Backticks, kein zusätzlicher Text.

AUSGABE-FELDER (nur diese Keys, exakt):
{
  "coverage": "gut|mittel|schlecht",
  "term_precision": "gut|mittel|schlecht",
  "error_level": "gut|mittel|schlecht",
  "wissensluecke": "gut|mittel|schlecht",
  "begruendung": "max. 2 Sätze, kurz und konkret"
}

REGELN ZUR WAHL DER KLASSEN:
- coverage:
  - gut: alle oder fast alle Kerninhalte aus dem Erwartungshorizont sind vorhanden
  - mittel: nur ein Teil der Kerninhalte ist vorhanden
  - schlecht: inhaltlich weitgehend verfehlt oder kein relevanter Bezug
- term_precision:
  - gut: Fachbegriffe und Bezeichnungen sind korrekt
  - mittel: kleinere Ungenauigkeiten, aber insgesamt erkennbar richtig
  - schlecht: wichtige Begriffe fehlen, sind falsch oder irreführend
- error_level:
  - gut: keine oder praktisch keine sachlichen Fehler
  - mittel: einzelne relevante Fehler oder Unschärfen, aber Kernaussage noch erkennbar
  - schlecht: mehrere oder schwere Fehler, die die Aussage deutlich verfälschen
- wissensluecke:
  - gut: solides Verständnis, keine wesentlichen Lücken
  - mittel: teilweise Verständnis, aber erkennbare Lücken bei wichtigen Aspekten
  - schlecht: grundlegendes Verständnis fehlt oder nur Bruchstücke sind erkennbar
"""

# ===== Hilfsfunktionen =====

def _extract_qa_from_messages(obj: Dict[str, Any]) -> Tuple[str, str]:
    msgs = obj.get("messages", [])
    q = ""
    a = ""
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user" and not q:
            q = (m.get("content") or "").strip()
        if m.get("role") == "assistant" and not a:
            a = (m.get("content") or "").strip()
    return q, a


def _first_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def map_judge_to_note(j: Dict[str, Any]) -> Tuple[int, int]:
    LEVEL_POINTS = {
        "gut": 5,
        "mittel": 3,
        "schlecht": 0,
    }

    coverage = j.get("coverage", "schlecht")
    term_precision = j.get("term_precision", "schlecht")
    error_level = j.get("error_level", "schlecht")
    wissensluecke = j.get("wissensluecke", "schlecht")

    c = LEVEL_POINTS.get(coverage, 0)
    t = LEVEL_POINTS.get(term_precision, 0)
    e = LEVEL_POINTS.get(error_level, 0)
    w = LEVEL_POINTS.get(wissensluecke, 0)

    total = c + t + e + w

    if total >= 16:
        punkte = 5
    elif total >= 8:
        punkte = 3
    else:
        punkte = 0

    note = 6 - punkte
    
    return note, punkte


def generate_candidate_answer(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": CANDIDATE_SYSTEM},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][prompt_len:]
    ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return ans if ans else "(keine Antwort generiert)"


def judge_answer(
    judge_model,
    judge_tokenizer,
    question: str,
    expected: str,
    candidate: str
) -> Tuple[str, Dict[str, Any], int, int]:
    user_payload = (
        f"FRAGE:\n{question}\n\n"
        f"ERWARTUNGSHORIZONT (Ground Truth):\n{expected}\n\n"
        f"MODELL-ANTWORT:\n{candidate}\n"
    )

    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    prompt = judge_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = judge_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=judge_tokenizer.eos_token_id,
            pad_token_id=judge_tokenizer.pad_token_id,
        )

    gen_ids = out[0][prompt_len:]
    raw = judge_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    js = _first_json_object(raw) or ""
    parsed = _safe_json_loads(js)

    if parsed is None:
        parsed = {
            "coverage": "schlecht",
            "term_precision": "schlecht",
            "error_level": "schlecht",
            "wissensluecke": "schlecht",
            "begruendung": "Judge-Ausgabe war kein gültiges JSON.",
        }

    note, punkte = map_judge_to_note(parsed)
    return raw, parsed, note, punkte


def append_model_result(
    fout,
    flog,
    tax_name: str,
    idx: int,
    model_name: str,
    question: str,
    expected: str,
    candidate: str,
    raw: str,
    parsed: Dict[str, Any],
    note: int,
    punkte: int
) -> None:
    record = {
        "taxonomie": tax_name,
        "index": idx,
        "model": model_name,
        "question": question,
        "expected": expected,
        "candidate": candidate,
        "judge_raw": raw,
        "judge": parsed,
        "note": note,
        "punkte": punkte,
    }
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    log_rec = {
        "taxonomie": tax_name,
        "index": idx,
        "model": model_name,
        "note": note,
        "punkte": punkte,
        "coverage": parsed.get("coverage"),
        "term_precision": parsed.get("term_precision"),
        "error_level": parsed.get("error_level"),
        "wissensluecke": parsed.get("wissensluecke"),
    }
    flog.write(json.dumps(log_rec, ensure_ascii=False) + "\n")


def plot_taxonomy_question_curves(
    tax_name: str,
    base_points: List[int],
    ft_points: List[int],
) -> None:
    x_base = list(range(1, len(base_points) + 1))
    x_ft = list(range(1, len(ft_points) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x_base, base_points, marker="o", label="Base")
    plt.plot(x_ft, ft_points, marker="o", label="Fine-Tuned")
    plt.xlabel("Frage-Index")
    plt.ylabel("Punkte")
    plt.title(f"Punkte-Verlauf pro Frage - {tax_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{tax_name}_points_curve.png"), dpi=150)
    plt.close()


def plot_taxonomy_averages(summary: Dict[str, Any]) -> None:
    taxonomies = [
        tax for tax in summary["base_model"].keys()
        if summary["base_model"][tax]["avg_punkte"] is not None
        and summary["fine_tuned_model"][tax]["avg_punkte"] is not None
    ]

    if not taxonomies:
        return

    base_avg_points = [
        summary["base_model"][tax]["avg_punkte"] for tax in taxonomies
    ]
    ft_avg_points = [
        summary["fine_tuned_model"][tax]["avg_punkte"] for tax in taxonomies
    ]

    x = list(range(len(taxonomies)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], base_avg_points, width=width, label="Base")
    plt.bar([i + width / 2 for i in x], ft_avg_points, width=width, label="Fine-Tuned")
    plt.xticks(x, taxonomies, rotation=45, ha="right")
    plt.ylabel("Durchschnittspunkte")
    plt.title("Durchschnittspunkte pro Taxonomie")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "taxonomy_avg_points.png"), dpi=150)
    plt.close()

def plot_overall_results(summary: Dict[str, Any]) -> None:
    labels = ["Base", "Fine-Tuned"]
    avg_points = [
        summary["gesamt"]["base_model"]["avg_punkte"],
        summary["gesamt"]["fine_tuned_model"]["avg_punkte"],
    ]

    if any(v is None for v in avg_points):
        return

    plt.figure(figsize=(8, 5))
    plt.bar(labels, avg_points)
    plt.ylabel("Durchschnittspunkte")
    plt.title("Gesamtergebnisse - Durchschnittspunkte")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "overall_avg_points.png"), dpi=150)
    plt.close()


# ===== Modelle laden =====
print("🚀 Lade Candidate Base-only ...")
base_tokenizer = AutoTokenizer.from_pretrained(
    CANDIDATE_BASE_MODEL_PATH,
    local_files_only=True,
    use_fast=True,
    trust_remote_code=True,
)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

base_model = Mistral3ForConditionalGeneration.from_pretrained(
    CANDIDATE_BASE_MODEL_PATH,
    device_map={"": 0} if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
    trust_remote_code=True,
)
base_model.eval()
if not torch.cuda.is_available():
    base_model.to("cpu")

print("🚀 Lade Candidate Fine-Tuned (Base + LoRA) ...")
ft_tokenizer = AutoTokenizer.from_pretrained(
    CANDIDATE_LORA_PATH,
    local_files_only=True,
    use_fast=True,
)
if ft_tokenizer.pad_token is None:
    ft_tokenizer.pad_token = ft_tokenizer.eos_token

ft_model = Mistral3ForConditionalGeneration.from_pretrained(
    CANDIDATE_BASE_MODEL_PATH,
    device_map={"": 0} if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(
    ft_model,
    CANDIDATE_LORA_PATH,
    local_files_only=True
)
ft_model.eval()
if not torch.cuda.is_available():
    ft_model.to("cpu")

print("🚀 Lade Judge (ohne LoRA) ...")
judge_tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL_PATH,
    local_files_only=True,
    use_fast=True,
    trust_remote_code=True,
)
if judge_tokenizer.pad_token is None:
    judge_tokenizer.pad_token = judge_tokenizer.eos_token

judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_PATH,
    device_map={"": 0} if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
    trust_remote_code=True,
)
judge_model.eval()
if not torch.cuda.is_available():
    judge_model.to("cpu")

print("✅ Modelle geladen.\n")

# ===== Main =====
summary: Dict[str, Any] = {
    "base_model": {},
    "fine_tuned_model": {},
    "vergleich": {}
}

paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.jsonl")))

if not paths:
    raise SystemExit(f"❌ Keine .jsonl Dateien gefunden in: {TEST_DIR}")

all_base_notes: List[int] = []
all_base_points: List[int] = []
all_ft_notes: List[int] = []
all_ft_points: List[int] = []

for fp in paths:
    tax_name = os.path.splitext(os.path.basename(fp))[0]

    out_base_path = os.path.join(OUT_DIR, f"{tax_name}_judge_base.jsonl")
    out_ft_path = os.path.join(OUT_DIR, f"{tax_name}_judge_ft.jsonl")

    log_base_path = os.path.join(LOG_DIR, f"{tax_name}_scores_base.jsonl")
    log_ft_path = os.path.join(LOG_DIR, f"{tax_name}_scores_ft.jsonl")

    base_notes: List[int] = []
    base_points: List[int] = []
    ft_notes: List[int] = []
    ft_points: List[int] = []

    print(f"=== Bewertet: {tax_name} ===")

    with open(fp, "r", encoding="utf-8") as fin, \
         open(out_base_path, "w", encoding="utf-8") as fout_base, \
         open(out_ft_path, "w", encoding="utf-8") as fout_ft, \
         open(log_base_path, "w", encoding="utf-8") as flog_base, \
         open(log_ft_path, "w", encoding="utf-8") as flog_ft:

        for idx, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            question, expected = _extract_qa_from_messages(obj)
            if not question:
                continue

            # ===== Base Model =====
            candidate_base = generate_candidate_answer(base_model, base_tokenizer, question)
            raw_base, parsed_base, note_base, punkte_base = judge_answer(
                judge_model,
                judge_tokenizer,
                question,
                expected,
                candidate_base
            )

            append_model_result(
                fout_base,
                flog_base,
                tax_name,
                idx,
                "base",
                question,
                expected,
                candidate_base,
                raw_base,
                parsed_base,
                note_base,
                punkte_base
            )

            base_notes.append(note_base)
            base_points.append(punkte_base)
            all_base_notes.append(note_base)
            all_base_points.append(punkte_base)

            # ===== Fine-Tuned Model =====
            candidate_ft = generate_candidate_answer(ft_model, ft_tokenizer, question)
            raw_ft, parsed_ft, note_ft, punkte_ft = judge_answer(
                judge_model,
                judge_tokenizer,
                question,
                expected,
                candidate_ft
            )

            append_model_result(
                fout_ft,
                flog_ft,
                tax_name,
                idx,
                "fine_tuned",
                question,
                expected,
                candidate_ft,
                raw_ft,
                parsed_ft,
                note_ft,
                punkte_ft
            )

            ft_notes.append(note_ft)
            ft_points.append(punkte_ft)
            all_ft_notes.append(note_ft)
            all_ft_points.append(punkte_ft)

    n_base = len(base_notes)
    n_ft = len(ft_notes)

    avg_base_note = sum(base_notes) / n_base if n_base else None
    avg_base_points = sum(base_points) / n_base if n_base else None

    avg_ft_note = sum(ft_notes) / n_ft if n_ft else None
    avg_ft_points = sum(ft_points) / n_ft if n_ft else None

    summary["base_model"][tax_name] = {
        "count": n_base,
        "avg_note": avg_base_note,
        "avg_punkte": avg_base_points,
        "out_file": out_base_path,
        "log_file": log_base_path,
    }

    summary["fine_tuned_model"][tax_name] = {
        "count": n_ft,
        "avg_note": avg_ft_note,
        "avg_punkte": avg_ft_points,
        "out_file": out_ft_path,
        "log_file": log_ft_path,
    }

    if avg_base_points is not None and avg_ft_points is not None:
        summary["vergleich"][tax_name] = {
            "delta_punkte_ft_minus_base": avg_ft_points - avg_base_points,
            "besseres_modell_punkte": (
                "fine_tuned" if avg_ft_points > avg_base_points
                else "base" if avg_base_points > avg_ft_points
                else "gleich"
            ),
        }

        plot_taxonomy_question_curves(
            tax_name=tax_name,
            base_points=base_points,
            ft_points=ft_points,
        )

        print(
            f"✔ base: count={n_base} | avg_punkte={avg_base_points:.3f}"
        )
        print(
            f"✔ ft:   count={n_ft} | avg_punkte={avg_ft_points:.3f}"
        )
        print(
            f"✔ delta(ft-base): punkte={avg_ft_points - avg_base_points:.3f}\n"
        )
    else:
        print("❌ Keine gültigen Beispiele gefunden.\n")

# ===== Gesamtauswertung =====
global_base_count = len(all_base_points)
global_ft_count = len(all_ft_points)

global_base_avg_points = sum(all_base_points) / global_base_count if global_base_count else None
global_ft_avg_points = sum(all_ft_points) / global_ft_count if global_ft_count else None

summary["gesamt"] = {
    "base_model": {
        "count": global_base_count,
        "avg_punkte": global_base_avg_points,
    },
    "fine_tuned_model": {
        "count": global_ft_count,
        "avg_punkte": global_ft_avg_points,
    },
    "vergleich": {
        "delta_punkte_ft_minus_base": (
            global_ft_avg_points - global_base_avg_points
            if global_ft_avg_points is not None and global_base_avg_points is not None
            else None
        ),
        "besseres_modell_punkte": (
            "fine_tuned" if global_ft_avg_points is not None and global_base_avg_points is not None and global_ft_avg_points > global_base_avg_points
            else "base" if global_ft_avg_points is not None and global_base_avg_points is not None and global_base_avg_points > global_ft_avg_points
            else "gleich"
        ),
    }
}

# ===== Summary schreiben =====
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

plot_taxonomy_averages(summary)
plot_overall_results(summary)

print("✅ Fertig.")
print("Summary:", summary_path)