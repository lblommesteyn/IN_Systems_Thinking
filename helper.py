import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm

# ---------- configuration ----------
MODEL_NAME = "google/flan-t5-small"   # tiny, instruction-tuned
PROMPT = (
    "Task: Does the paragraph describe a *cross-scale interaction* (CSI)?\n"
    "Definition: A CSI links processes at *different* spatial or temporal scales "
    "(e.g., global ↔︎ local, long-term ↔︎ short-term) or describes feedback between them.\n"
    "Respond with just '1' (yes) or '0' (no).\n\n"
    "Paragraph:\n"
)
CSV_IN  = "all_nestle_unilever_paragraphs.csv"
CSV_OUT = "all_nestle_unilever_paragraphs_with_CSI_LLMSmall.csv"
BATCH   = 16   # tweak for your hardware
# -----------------------------------

print("Loading model…")
clf = pipeline(
    "text2text-generation",
    model=MODEL_NAME,
    max_new_tokens=2,
    batch_size=BATCH,
    device_map="auto"       # GPU if available, else CPU
)

df = pd.read_csv(CSV_IN)
paragraphs = df["paragraph"].tolist()
preds = []

print("Classifying paragraphs…")
for i in tqdm(range(0, len(paragraphs), BATCH)):
    batch = [PROMPT + p for p in paragraphs[i:i+BATCH]]
    outs = clf(batch)
    for out in outs:
        answer = out["generated_text"].strip()
        preds.append(1 if answer.startswith("1") else 0)

df["CSI_LLMSmall"] = preds
df.to_csv(CSV_OUT, index=False)
print(f"Done → {CSV_OUT}")
