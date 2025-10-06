import pandas as pd

def get_relationship_type(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    t = text.lower()
    causal_kw = ["leads to", "drives", "because"]
    correlational_kw = ["linked to", "associated with"]
    feedback_kw = ["reinforces", "loop", "in turn"]
    for kw in causal_kw:
        if kw in t:
            return "Causal"
    for kw in correlational_kw:
        if kw in t:
            return "Correlational"
    for kw in feedback_kw:
        if kw in t:
            return "Feedback"
    if t.count('→') >= 2:
        return "Feedback"
    return "Unknown"


def main():
    input_path = r"c:\Users\16476\OneDrive\Desktop\IN_Systems thinking\Unilever_Nestle_testing\v3_paragraph_results.csv"
    output_path = r"c:\Users\16476\OneDrive\Desktop\IN_Systems thinking\Unilever_Nestle_testing\v4_paragraph_results_with_types.csv"

    df = pd.read_csv(input_path)
    for col in df.columns:
        if col.strip().endswith("Relationship"):
            df[f"{col} Type"] = df[col].apply(get_relationship_type)
    df.to_csv(output_path, index=False)
    print(f"Created file with types: {output_path}")

if __name__ == "__main__":
    main()
