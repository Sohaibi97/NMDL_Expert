#!/usr/bin/env python3
# judge_answers.py
#
# Erwartet Testdaten in:  Test_data/*.jsonl
# Jede Zeile: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
# => "assistant" wird als Erwartungshorizont (Referenz) genutzt.
# Das Script generiert eine Candidate-Antwort mit deinem finetuned Modell
# und lässt ein separates Judge-Modell streng bewerten.
#
# Outputs:
# - judge_outputs/<Taxonomie>_judge.jsonl    (voller Record inkl. Judge-JSON + Note/Punkte)
# - judge_logs/<Taxonomie>_scores.jsonl      (kompakte Logs je Frage: Note/Punkte + Klassifikationen)
# - judge_outputs/summary.json               (Durchschnittsnote je Taxonomie)

import os
import json
import glob
import re
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Offline erzwingen (HPC ohne Internet) ---
os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

# ===== Pfade (Wo die Testdaten und Ergebnisse gespeichert werden) =====
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

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===== Candidate System Prompt (für Generierung) =====
CANDIDATE_SYSTEM = os.environ.get(
    "CANDIDATE_SYSTEM",
    "Du bist ein Experte für NMDL. Beantworte präzise und fachlich korrekt."
)

# ===== Strenger Judge Prompt (Klassifikationen + JSON-only) =====
JUDGE_PROMPT = """Du bist ein strenger, objektiver Bewertungsassistent.
Du vergleichst eine MODELL-ANTWORT mit einem ERWARTUNGSHORIZONT (Ground Truth).
Bewertet wird ausschließlich die MODELL-ANTWORT: sachliche Korrektheit und Abdeckung der Kerninhalte.
Stil, Länge, Höflichkeit sind irrelevant.

WICHTIG:
- Inhalt vor Wortlaut: Paraphrasen sind OK, solange der Sachverhalt korrekt ist.
- Aber: Fachbegriffe, Eigennamen und definierte Bezeichnungen müssen präzise korrekt sein.
- Wissensluecke: beschreibt die Stärke der Lücke; "keine" bedeutet keine Lücke (bestmöglich), "erhebliche" bedeutet starke Lücke (schlechtestmöglich).
- Konkrete falsche Behauptungen spiegeln sich in error_level ("einige" oder "dominant") wider.

ZUSATZINFORMATIONEN:
- Korrekte Zusatzinformationen dürfen NIEMALS die coverage oder error_level verschlechtern.
- Setze coverage="voll", sobald alle Kerninhalte aus dem Erwartungshorizont vorhanden sind – auch wenn mehr erklärt wird.
- Setze error_level nur bei tatsächlichen sachlichen Fehlern oder Widersprüchen (nicht bei „zu viel Kontext“).
- Unklare Zusatzbegriffe (z.B. neue Abkürzungen) sind nur dann negativ, wenn sie dem Erwartungshorizont widersprechen.

Du MUSST GENAU EIN JSON-OBJEKT ausgeben, nichts anderes.
Kein Markdown, keine Backticks, kein zusätzlicher Text.

AUSGABE-FELDER (nur diese Keys, exakt):
{
  "coverage": "voll|ueberwiegend|teilweise|einzelne|kein_bezug",
  "term_precision": "korrekt|leicht_ungenau|fehlerhaft_oder_fehlend",
  "error_level": "keine|klein|einige|dominant",
  "wissensluecke": "keine|kleine|grosse|erhebliche",
  "begruendung": "max. 2 Sätze, kurz und konkret"
}

REGELN ZUR WAHL DER KLASSEN:
- coverage:
  - voll: alle Kerninhalte aus Erwartung vorhanden
  - ueberwiegend: fast alles vorhanden, kleine Lücken
  - teilweise: einige Kerninhalte fehlen
  - einzelne: nur wenige Elemente erkennbar
  - kein_bezug: thematisch verfehlt / kein Bezug
- term_precision:
  - korrekt: Fachbegriffe/Bezeichnungen korrekt
  - leicht_ungenau: kleine Ungenauigkeiten, aber erkennbar korrekt
  - fehlerhaft_oder_fehlend: wichtige Begriffe falsch oder fehlen
- error_level:
  - keine: keine faktischen Fehler
  - klein: kleine Fehler/Unschärfen, Kernaussage korrekt
  - einige: mehrere Fehler oder relevante Fehler
  - dominant: Fehler dominieren, Aussage überwiegend falsch
- wissensluecke:
  - keine: Antwort zeigt solides Verständnis, keine erkennbaren Lücken
  - kleine: Randaspekte fehlen oder sind oberflächlich, Kernverständnis vorhanden
  - grosse: wesentliche Konzepte fehlen oder werden falsch verstanden
  - erhebliche: fundamentales Verständnis fehlt, nur Bruchstücke erkennbar
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
    # minimal robust: nimm vom ersten "{" bis zum letzten "}"
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
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def map_judge_to_note(j: Dict[str, Any]) -> Tuple[int, int]:
    # Punktetabellen für die Klassifikationen
    COVERAGE_POINTS = {
        "voll": 5, "ueberwiegend": 4, "teilweise": 3, "einzelne": 2, "kein_bezug": 0
    }
    TERM_POINTS = {
        "korrekt": 5, "leicht_ungenau": 3, "fehlerhaft_oder_fehlend": 0
    }
    ERROR_POINTS = {
        "keine": 5, "klein": 3, "einige": 1, "dominant": 0
    }
    WISS_POINTS = {
        "keine": 5, "kleine": 3, "grosse": 1, "erhebliche": 0
    }

    coverage = j.get("coverage", "kein_bezug")
    term_precision = j.get("term_precision", "fehlerhaft_oder_fehlend")
    error_level = j.get("error_level", "dominant")
    wissensluecke = j.get("wissensluecke", "erhebliche")

    # Wenn kein Bezug: direkt 0 Punkte / Note 6
    if coverage == "kein_bezug":
        return 6, 0

    c = COVERAGE_POINTS.get(coverage, 0)
    t = TERM_POINTS.get(term_precision, 0)
    e = ERROR_POINTS.get(error_level, 0)
    w = WISS_POINTS.get(wissensluecke, 0)

    total = c + t + e + w
    total_max = 5 + 5 + 5 + 5

    punkte = int(round((total / total_max) * 5))
    punkte = max(0, min(5, punkte))

    note = 6 - punkte
    return note, punkte


def generate_candidate_answer(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": CANDIDATE_SYSTEM},
        {"role": "user", "content": question},
    ]
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][prompt_len:]
    ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return ans if ans else "(keine Antwort generiert)"


def judge_answer(judge_model, judge_tokenizer, question: str, expected: str, candidate: str) -> Tuple[str, Dict[str, Any], int, int]:
    # Judge sieht: Frage + Erwartung + Modellantwort
    user_payload = (
        f"FRAGE:\n{question}\n\n"
        f"ERWARTUNGSHORIZONT (Ground Truth):\n{expected}\n\n"
        f"MODELL-ANTWORT:\n{candidate}\n"
    )

    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    prompt = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        # Fallback: maximal streng => Note 6
        parsed = {
            "coverage": "kein_bezug",
        "term_precision": "fehlerhaft_oder_fehlend",
        "error_level": "dominant",
        "wissensluecke": "erhebliche",
        "begruendung": "Judge-Ausgabe war kein gültiges JSON.",
        }

    note, punkte = map_judge_to_note(parsed)
    return raw, parsed, note, punkte


# ===== Laden der Modelle =====
print("🚀 Lade Candidate (Base + LoRA) ...")
cand_tokenizer = AutoTokenizer.from_pretrained(CANDIDATE_LORA_PATH, local_files_only=True, use_fast=True)
if cand_tokenizer.pad_token is None:
    cand_tokenizer.pad_token = cand_tokenizer.eos_token

candidate_model = Mistral3ForConditionalGeneration.from_pretrained(
    CANDIDATE_BASE_MODEL_PATH,
    device_map={"": 0} if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    local_files_only=True,
    trust_remote_code=True,
)
candidate_model = PeftModel.from_pretrained(candidate_model, CANDIDATE_LORA_PATH, local_files_only=True)
candidate_model.eval()
if not torch.cuda.is_available():
    candidate_model.to("cpu")

print("🚀 Lade Judge (ohne LoRA) ...")
#judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH, local_files_only=True, use_fast=True)
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH, local_files_only=True, use_fast=True, trust_remote_code=True)
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

# ===== Main: alle Taxonomie-JSONL bewerten =====
summary: Dict[str, Any] = {}
paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.jsonl")))

if not paths:
    raise SystemExit(f"❌ Keine .jsonl Dateien gefunden in: {TEST_DIR}")

for fp in paths:
    tax_name = os.path.splitext(os.path.basename(fp))[0]
    out_path = os.path.join(OUT_DIR, f"{tax_name}_judge.jsonl")
    log_path = os.path.join(LOG_DIR, f"{tax_name}_scores.jsonl")

    notes: List[int] = []
    points: List[int] = []

    print(f"=== Bewertet: {tax_name} ===")
    with open(fp, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout, \
         open(log_path, "w", encoding="utf-8") as flog:

        for idx, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                # skip kaputte Zeile
                continue

            question, expected = _extract_qa_from_messages(obj)
            if not question:
                continue

            # 1) Candidate generieren
            candidate = generate_candidate_answer(candidate_model, cand_tokenizer, question)

            # 2) Judge bewerten (JSON-only)
            raw, parsed, note, punkte = judge_answer(judge_model, judge_tokenizer, question, expected, candidate)

            notes.append(note)
            points.append(punkte)

            record = {
                "taxonomie": tax_name,
                "index": idx,
                "question": question,
                "expected": expected,
                "candidate": candidate,
                "judge_raw": raw,
                "judge": parsed,
                "note": note,
                "punkte": punkte,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # kompakter Log (Score pro Frage)
            log_rec = {
                "taxonomie": tax_name,
                "index": idx,
                "note": note,
                "punkte": punkte,
                "coverage": parsed.get("coverage"),
                "term_precision": parsed.get("term_precision"),
                "error_level": parsed.get("error_level"),
                "wissensluecke": parsed.get("wissensluecke"),
            }
            flog.write(json.dumps(log_rec, ensure_ascii=False) + "\n")

    n = len(notes)
    avg_note = sum(notes) / n if n else None
    avg_points = sum(points) / n if n else None

    summary[tax_name] = {
        "count": n,
        "avg_note": avg_note,
        "avg_punkte": avg_points,
        "out_file": out_path,
        "log_file": log_path,
    }

    if avg_note is None:
        print("❌ Keine gültigen Beispiele gefunden.\n")
    else:
        print(f"✔ count={n} | avg_note={avg_note:.3f} | avg_punkte={avg_points:.3f}\n")

# Summary schreiben
summary_path = os.path.join(OUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("✅ Fertig.")
print("Summary:", summary_path)