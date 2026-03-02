#!/usr/bin/env python3
import torch
from typing import cast, Any, Union, Sequence
from datasets import Dataset, load_dataset
from transformers import (
    Mistral3ForConditionalGeneration, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl.trainer.sft_trainer import SFTTrainer

# ===== 1) Robuster Collator =====
class NMDLCompletionCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, mlm=False, *args, **kwargs)
        self.response_template = response_template

    def torch_call(self, examples: Sequence[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        new_examples = []
        for ex in examples:
            ids = ex["input_ids"]
            # Extraktion aus komplexen Mistral-Objekten
            if hasattr(ids, "input_ids"): ids = ids["input_ids"]
            if hasattr(ids, "ids"): ids = ids.ids
            if torch.is_tensor(ids): ids = ids.tolist()
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list): ids = ids[0]
            
            new_examples.append({"input_ids": [int(x) for x in ids]})
        
        batch = super().torch_call(new_examples)
        for i in range(len(batch["input_ids"])):
            token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
            input_ids = batch["input_ids"][i].tolist()
            idx = -1
            for j in range(len(input_ids) - len(token_ids) + 1):
                if input_ids[j : j + len(token_ids)] == token_ids:
                    idx = j + len(token_ids)
                    break
            if idx != -1:
                batch["labels"][i, :idx] = -100
        return batch

# ===== 2) Konfiguration =====
MODEL_ID = "/home/khamlichi/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512-BF16/snapshots/ecc3ba8b43a45610e709327c049d24b009bfec88"
DATA_FILE = "/home/khamlichi/Projekt_NMDL_2/Data/data.jsonl"
OUT_DIR = "/home/khamlichi/Projekt_NMDL_2/outputs/ministral3b-instruct-nmdl-lora"
SYSTEM_MESSAGE = "Du bist ein Experte für NMDL. Beantworte alle Fragen präzise."

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

ds = cast(Dataset, load_dataset("json", data_files=DATA_FILE, split="train"))
ds = ds.map(lambda ex: {"messages": ([{"role": "system", "content": SYSTEM_MESSAGE}] + ex["messages"]) if ex["messages"][0]["role"] != "system" else ex["messages"]})

# ===== 3) Modell & LoRA =====
model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True
)
model.gradient_checkpointing_enable()

model = cast(PeftModel, get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)))

# ===== 4) Training Arguments =====
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=4,
    logging_steps=10,
    bf16=True,
    report_to="none",
    remove_unused_columns=False
)

# ===== 5) Start =====
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=NMDLCompletionCollator(response_template="[/INST]", tokenizer=tokenizer),
    processing_class=tokenizer,
)

print("--- Start Training (Finaler HPC Fix) ---")
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)