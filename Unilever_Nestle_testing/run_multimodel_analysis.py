#!/usr/bin/env python3
"""
Run multi-model analysis on paragraphs and write results to CSV.
"""

import config
import argparse
import pandas as pd
import openai
import anthropic
import google.generativeai as genai
import json

# Setup API keys
openai.api_key = config.OPENAI_API_KEY
claude_client = anthropic.Client(api_key=config.ANTHROPIC_API_KEY)
genai.configure(api_key=config.GOOGLE_API_KEY)

# Unified improved prompt combining best of version1 and version2
PROMPT = r"""
Rule Book v3: Identifying Cross-Scale Connections in Annual Reports

0. Entity Ontology & Tags:
  - ORG: Organisation (company, NGO, UN, regulator)
  - PPL: People/Groups (employees, executives, communities)
  - IND: Industry/Sector (manufacturing sector, renewable energy)
  - ENV: Environmental/Socio-economic system (climate system, ecosystems, financial markets)
  - GEO: Geography (Europe, coastal regions, Asia-Pacific)
  - TMP: Temporal actor (future, long-term strategies)

1. Relationship Keyword Hints:
  - Causal: leads to, results in, causes, drives, because, due to, fosters
  - Correlational: associated with, linked to, correlated with, coincides with
  - Feedback: in turn, cycle, loop, feedback, recursively, reinforces, dampens

2. Numeric Rubric for Strength:
  - 5-Strong: explicit causal verbs + both scales named + systemic factor
  - 4-Above-Avg: explicit causal verbs + at least one scale difference clear
  - 3-Moderate: causal verbs implicit or partial; some inference needed
  - 2-Weak: vague connection, aspirational wording
  - 1-Very-Weak: boilerplate, no discernible relationship

3. Systemic-Factor Glossary:
  - Climate system: greenhouse gases, flooding, drought, temperature rise
  - Ecosystems: biodiversity, fish stocks, forests
  - Financial system: capital markets, interest rates, credit risk
  - Regulatory system: policy, laws, compliance cycles
  - Socio-economic: inequality, labour markets, consumer confidence

4. Decision Flow:
  1. Does paragraph mention ≥2 entities? If No ⇒ cross_scale_connection: No (stop)
  2. Is there a causal/correlational link? If No ⇒ relationship_type: none
  3. Do cause & effect occur at different spatial OR temporal scales? If Yes ⇒ cross_scale_connection: Yes
  4. Check for feedback keywords; if present ⇒ feedback_loop_present: Yes and classify loop_nature (reinforcing/balancing)
  5. Record systemic_factor_involved if an ENV entity present or implied

5. Multi-Paragraph Guidance:
  - If context spans paragraphs, mentally merge and still apply rules; note in summary

6. Output Format:
  Return exactly one JSON object with keys:
    cross_scale_connection (Yes|No), feedback_loop_present (Yes|No),
    entities (list of TAG:entity), relationship_type (mention both entities), loop_nature,
    systemic_factor_involved, strength, explicitness, confidence (0-1), summary
  Return JSON only.
"""

# Map friendly tags to model identifiers
MODELS = {
    "o3": "gpt-3.5-turbo",
    "4o": "gpt-4o",
    "o4_mini_high": "gpt-4o-mini-high",
    "o4_mini": "gpt-4o-mini",
    "claude3.5": "claude-3.5",
    "claude3.7": "claude-3.7",
    "gemini1": "gemini-1.0",
    "gemini2": "gemini-pro"
}

def analyze_openai(model, text):
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text.strip()}
        ]
    )
    return resp.choices[0].message.content

def analyze_claude(model, text):
    res = claude_client.completions.create(
        model=model,
        prompt=PROMPT + "\n" + text.strip(),
        max_tokens_to_sample=600,
        temperature=0,
        format="json"
    )
    return res.completion

def analyze_gemini(model, text):
    res = genai.chat.completions.create(
        model=model,
        prompt=PROMPT + "\n" + text.strip(),
        temperature=0,
        max_output_tokens=600
    )
    return res.candidates[0].content

def parse_json(content):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return eval(content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="paragraphs_comparison.csv",
                        help="Input CSV with paragraphs and metadata")
    parser.add_argument("--output", "-o", default="multimodel_results.csv",
                        help="Output CSV with multi-model results")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    tags = list(MODELS.keys())
    results = []

    for _, row in df.iterrows():
        text = row.get("Full Paragraph Text", "")
        rec = {
            "Full Paragraph Text": text,
            "Source": row.get("Source", ""),
            "Entities": row.get("Entities", "")
        }
        identified = []
        for tag, model in MODELS.items():
            try:
                if tag.startswith("claude"):
                    out = parse_json(analyze_claude(model, text))
                elif tag.startswith("gemini"):
                    out = parse_json(analyze_gemini(model, text))
                else:
                    out = parse_json(analyze_openai(model, text))
                rec[f"{tag}: Found"] = out.get("cross_scale_connection")
                rec[f"{tag}: Relationship"] = out.get("relationship_type")
                rec[f"{tag}: Cross-Scale"] = out.get("cross_scale_connection")
                rec[f"{tag}: Feedback Loop"] = out.get("feedback_loop_present")
                rec[f"{tag}: Systemic"] = out.get("systemic_factor_involved")
                rec[f"{tag}: Strength"] = out.get("strength")
                rec[f"{tag}: Explicit"] = out.get("explicitness")
                rec[f"{tag}: Confidence"] = out.get("confidence")
                rec[f"{tag}: Summary"] = out.get("summary")
                if out.get("cross_scale_connection") == "Yes":
                    identified.append(tag)
            except Exception:
                for field in ["Found","Relationship","Cross-Scale","Feedback Loop","Systemic","Strength","Explicit","Confidence","Summary"]:
                    rec[f"{tag}: {field}"] = None
        rec["Models Identified"] = len(identified)
        rec["Models List"] = ", ".join(identified)
        # detect disagreements
        fields = ["Found","Relationship","Cross-Scale","Feedback Loop","Systemic","Strength","Explicit","Confidence","Summary"]
        diff = []
        for field in fields:
            vals = {rec.get(f"{tag}: {field}") for tag in tags}
            if len(vals) > 1:
                diff.append(field)
        rec["Has Disagreements"] = "Yes" if diff else "No"
        rec["Disagreement Fields"] = ", ".join(diff)
        results.append(rec)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote results to {args.output}")

if __name__ == "__main__":
    main()
