# Cross‑Scale Connections Notebook v2 (using LLM + validation)

"""
Notebook Objective (v2)
=======================
Analyse any set of paragraphs (e.g. annual‑report text) and tag them for cross‑scale connections **with tighter rules, rubric, ontology, and JSON validation**.

Improvements added in v2
-----------------------
1. **Entity Ontology** with ID tags (ORG, PPL, IND, ENV, GEO, TMP).
2. **Keyword cheat‑sheets** for causal / correlational / feedback triggers.
3. **Positive & negative examples** embedded in prompt.
4. **Decision flowchart** (as plain text instructions).
5. **Numeric rubric** for strength & explicitness.
6. **Confidence score** requested from the LLM.
7. **Systemic‑factor glossary** (Climate, Ecosystems, Financial, Regulatory, Socio‑economic).
8. **Multi‑paragraph guidance** (merge rule).
9. **JSON‑Schema validation** with automatic repair attempt.

"""

# --- Configuration -----------------------------------------------------------
import os, json
import openai, pandas as pd
from typing import List, Dict
from jsonschema import validate, ValidationError

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")  # ← set or export
MODEL = "gpt-4o-mini"  # or any ChatCompletion‑capable model

# --- JSON Schema -------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "required": [
        "cross_scale_connection", "feedback_loop_present", "entities",
        "relationship_type", "loop_nature", "systemic_factor_involved",
        "scale_differences", "strength", "explicitness", "confidence", "summary"
    ],
    "properties": {
        "cross_scale_connection": {"enum": ["Yes", "No"]},
        "feedback_loop_present":  {"enum": ["Yes", "No"]},
        "entities": {"type": "array", "items": {"type": "string"}},
        "relationship_type": {"enum": ["causal", "correlational", "feedback", "mixed", "none"]},
        "loop_nature": {"enum": ["reinforcing", "balancing", "n/a"]},
        "systemic_factor_involved": {"enum": ["Yes", "No"]},
        "scale_differences": {
            "type": "object",
            "properties": {"spatial": {"type": "boolean"}, "temporal": {"type": "boolean"}},
            "required": ["spatial", "temporal"]
        },
        "strength": {"enum": ["5-Strong", "4‑Above‑Avg", "3‑Moderate", "2‑Weak", "1‑Very‑Weak"]},
        "explicitness": {"enum": ["Explicit", "Implicit"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"}
    }
}

# --- Rulebook (full v2 prompt) ----------------------------------------------
RULEBOOK_PROMPT = r"""
Rule Book v2: Identifying Cross‑Scale Connections in Annual Reports
==================================================================

### 0. Entity Ontology & Tags
| Tag | Category | Examples |
|-----|----------|----------|
| ORG | Organisation | company, NGO, UN, regulator |
| PPL | People / Groups | employees, executives, communities |
| IND | Industry / Sector | manufacturing sector, renewable energy |
| ENV | Environmental or Socio‑Economic System | climate system, ecosystems, financial markets |
| GEO | Geography | Europe, coastal regions, Asia‑Pacific |
| TMP | Temporal Actor | future, long‑term strategies |

Return entities as "<TAG>:<surface‑text>" (e.g. "ORG:Unilever").

### 1. Relationship Keyword Hints
Causal: leads to, results in, causes, drives, because, due to, fosters →
Correlational: associated with, linked to, correlated with, coincides with
Feedback indicators: in turn, cycle, loop, feedback, recursively, reinforces, dampens

### 2. Positive & Negative Examples
*Positive:* "Investments in climate resilience reduce risks, encouraging more investments in resilience."  → Reinforcing feedback across firm ↔ climate ↔ firm.
*Negative (looks tempting but **fails**):* "We continue to prioritise innovation to delight customers." (No cross‑scale relation, no systemic factor.)

### 3. Decision Flow (follow sequentially)
1. Does paragraph mention ≥2 entities? If **No** ⇒ tag `cross_scale_connection: No` and stop.
2. Is there a causal/correlational link? If **No** ⇒ tag `relationship_type:none`.
3. Do cause & effect occur at different spatial **or** temporal scales? If **Yes** ⇒ `cross_scale_connection: Yes`.
4. Check feedback pattern keywords; if present => `feedback_loop_present: Yes` and classify loop nature.
5. Record systemic factor (ENV tagged) if present or logically necessary.

### 4. Numeric Rubric (Strength)
5‑Strong   = explicit causal verbs **+** both scales named **+** systemic factor.
4‑Above‑Avg= explicit causal verbs **+** at least one scale difference clear.
3‑Moderate = causal verbs implicit or partial; some inference needed.
2‑Weak     = vague connection, aspirational wording.
1‑Very‑Weak= boilerplate, no discernible relationship.

Explicitness = *Explicit* if causal wording present; otherwise *Implicit*.

### 5. Systemic‑Factor Glossary
*Climate system*: greenhouse gases, flooding, drought, temperature rise.
*Ecosystems*: biodiversity, fish stocks, forests.
*Financial system*: capital markets, interest rates, credit risk.
*Regulatory system*: policy, laws, compliance cycles.
*Socio‑economic*: inequality, labour markets, consumer confidence.

### 6. Multi‑Paragraph Guidance
If the paragraph relies on the **previous** one for cause or effect, merge them mentally and still apply rules; output field `merged_context` in summary if used.

### 7. Output JSON
Return exactly one JSON object matching the schema provided separately (see conversation context). Include:
* `confidence` = your self‑estimated probability (0‑1) that the tagging is correct.
"""

# --- LLM Helper --------------------------------------------------------------

def llm_analyze(paragraph: str) -> Dict:
    """Send paragraph to LLM, validate JSON, attempt auto‑repair once if invalid."""
    def _call(msg: str):
        return openai.ChatCompletion.create(
            model=MODEL,
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": RULEBOOK_PROMPT},
                {"role": "assistant", "content": json.dumps({"schema": SCHEMA})},
                {"role": "user", "content": msg.strip()}
            ]
        ).choices[0].message.content

    result_json = _call(paragraph)

    # validate & repair once
    try:
        parsed = json.loads(result_json)
        validate(instance=parsed, schema=SCHEMA)
        return parsed
    except (ValidationError, json.JSONDecodeError):
        # one repair attempt by asking model to fix output
        fix_prompt = f"The previous JSON was invalid: {result_json}. Please output valid JSON matching the schema exactly."
        fixed = _call(fix_prompt)
        try:
            parsed = json.loads(fixed)
            validate(instance=parsed, schema=SCHEMA)
            return parsed
        except Exception as e:
            raise ValueError(f"JSON validation failed after repair: {e}")

# --- Batch Utility -----------------------------------------------------------

def analyze_paragraphs(paragraphs: List[str]) -> pd.DataFrame:
    rows = []
    for p in paragraphs:
        try:
            res = llm_analyze(p)
            rows.append({"paragraph": p, **res})
        except Exception as err:
            rows.append({"paragraph": p, "error": str(err)})
    return pd.DataFrame(rows)

# --- Example -----------------------------------------------------------------
if __name__ == "__main__":
    test_paragraphs = [
        "Investments in climate resilience reduce risks, encouraging more investments in resilience.",
        "We continue to prioritise innovation to delight customers."
    ]
    df = analyze_paragraphs(test_paragraphs)
    print(df.to_markdown(index=False))
