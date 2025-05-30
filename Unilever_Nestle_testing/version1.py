# Cross-Scale Connections Notebook using LLM

"""
Notebook Objective:
Given a set of paragraphs (e.g., from annual reports), this notebook uses a Large Language Model (LLM) to classify and tag paragraphs according to the **full** rulebook on cross‑scale connections.

Fill in your OpenAI (or compatible) API key, run the helper, and feed it any text source (e.g., Unilever 2014 PDF split into paragraphs).
"""

# --- Configuration -----------------------------------------------------------
import os
import openai
import pandas as pd
from typing import List, Dict

openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")  # ← edit or set env‑var
MODEL = "gpt-4o-mini"  # pick any chat‑completion capable model

# --- Rulebook ----------------------------------------------------------------
RULEBOOK_PROMPT = r"""
Rule Book: Identifying Cross-Scale Connections in Annual Reports

Objective
Carefully analyze each paragraph to determine whether it describes relationships across different scales (spatial or temporal) — including feedback loops involving systemic factors (e.g., climate, ecosystems, financial systems).

Step 1. Identify all entities mentioned
Entities = any actor, system, or concept that can act, be acted upon, or mediate relationships:
• Organizations (e.g., "ABC Corp", "UN", "industry peers")
• People or groups (e.g., "executives", "communities", "future generations")
• Industries or sectors (e.g., "renewable energy", "manufacturing sector")
• Natural or social environments (e.g., "climate system", "ecosystems", "financial markets")
• Geographical regions (e.g., "Asia-Pacific", "coastal areas")
• Temporal actors (e.g., "future", "long‑term strategies")
Instructions:
List all entities mentioned explicitly or implicitly.
Be specific: identify entity type and scale if possible (e.g., "firm‑level investment", "global climate system").

Step 2. Identify all relationships among entities, including feedback loops
Types of relationships:
• Causal (A leads to B)
  – Example: "Increased production causes higher emissions."
• Correlational (A and B are linked)
  – Example: "ESG investments are associated with stronger market performance."
• Feedback loops (cyclical causality: A affects B, B affects A or another element)
  – Reinforcing (positive): A change amplifies further changes. Example: "Adoption of renewables lowers costs, which drives more adoption."
  – Balancing (negative): A change triggers actions that stabilize the system. Example: "Overfishing triggers stricter regulations, reducing overfishing."

Special Note on Feedback Loops:
Feedback loops often involve systemic mediators (e.g., climate, ecosystems, markets), even if they are not explicitly named.
Example: A report states, "increased coastal flooding leads to higher insurance claims, which in turn raises insurance premiums." Although the climate system is not explicitly mentioned, the reference to "coastal flooding" provides a clear basis to infer climate-related systemic involvement.
Infer system-level involvement only when there is a clear indication that broader impacts beyond the immediate entity are at play.
Example: A company notes, "declining fish stocks have reduced local fishing revenues, leading to economic hardship in coastal towns." Because "declining fish stocks" directly points to an ecosystem dynamic, it is appropriate to recognize ecosystem-level involvement without making further assumptions.
Always record if a systemic factor is clearly indicated or logically necessary for understanding the relationship.
Example: An organization reports, "our investment in renewable energy reduced our carbon emissions and improved stakeholder reputation." Since carbon emissions directly relate to the climate system, it is appropriate to record "climate system" as part of the feedback loop, even if not named outright.
Instructions:
Identify cause‑effect links, correlations, and feedback loops.
For feedback loops, specify:
  • Entities involved.
  • Nature of loop (Reinforcing or Balancing).
  • Whether a systemic mediator (e.g., climate) is involved (even implicitly).

Step 3. Determine if relationships cross different scales
Scales to check:
Spatial scale:
  Local → Regional → National → Global
  Individual → Team/Unit → Organization‑wide
Temporal scale:
  Immediate → Short‑term → Long‑term → Intergenerational
Special Instructions:
Within organizations, treat individual‑level, team‑level, and firm‑level phenomena as occurring at different scales.
When individuals' actions are linked to organizational outcomes, treat it as a cross‑scale connection (micro → macro).
Questions to ask:
  • Are the cause and effect happening at different spatial scales?
  • Are the cause and effect happening at different time horizons?
  • Is a systemic factor involved that connects multiple scales?

Step 4. Evaluate the strength and explicitness of the cross‑scale connections
Strength:
  • Strong: Direct, unambiguous causal or feedback relationships.
  • Moderate: Partially described, but link can be reasonably inferred.
  • Weak: Vague, aspirational, or generic statements without clear links.
Explicitness:
  • Explicit: The cross‑scale connection is clearly articulated.
  • Implicit: The cross‑scale connection must be inferred by the reader based on context.

Step 5. Final Tagging
For each paragraph output exactly this JSON schema:
{
  "cross_scale_connection": "Yes|No",
  "feedback_loop_present": "Yes|No",
  "entities": ["..."],
  "relationship_type": "causal|correlational|feedback|mixed|none",
  "loop_nature": "reinforcing|balancing|n/a",
  "systemic_factor_involved": "Yes|No",
  "scale_differences": {"spatial": true|false, "temporal": true|false},
  "strength": "Strong|Moderate|Weak",
  "explicitness": "Explicit|Implicit",
  "summary": "<concise 1‑2 sentence explanation>"
}
"""

# --- Helper ------------------------------------------------------------------

def llm_analyze(paragraph: str) -> Dict:
    """Send a single paragraph to the LLM and return the parsed JSON."""
    completion = openai.ChatCompletion.create(
        model=MODEL,
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RULEBOOK_PROMPT},
            {"role": "user", "content": paragraph.strip()}
        ]
    )
    return completion.choices[0].message.to_dict()["content"]

# --- Batch Utility -----------------------------------------------------------

def analyze_paragraphs(paragraphs: List[str]) -> pd.DataFrame:
    records = []
    for p in paragraphs:
        try:
            record_json = llm_analyze(p)
            records.append({"paragraph": p, **eval(record_json)})
        except Exception as e:
            records.append({"paragraph": p, "error": str(e)})
    return pd.DataFrame(records)

# --- Example -----------------------------------------------------------------
if __name__ == "__main__":
    sample = "Investments in climate resilience reduce risks, encouraging more investments in resilience."
    df = analyze_paragraphs([sample])
    print(df)
