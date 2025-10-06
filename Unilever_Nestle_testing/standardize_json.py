import json
import os
import glob

# Define the directory containing the JSON files
directory = os.path.dirname(os.path.abspath(__file__))

# Define the output directory for standardized files
output_directory = os.path.join(directory, "standardized_json")

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get all JSON files in the directory
json_files = glob.glob(os.path.join(directory, "*.json"))

# Define the standardized structure
# Based on analysis, we'll use a structure with a top-level array containing objects with all fields
# This matches the structure of 4o_paragraphs.json which has the most complete set of fields

# Function to standardize a JSON file
def standardize_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        standardized_data = []
        
        # Handle different structures
        if isinstance(data, list):
            # Already a list at the top level (like 4o_paragraphs.json)
            source_data = data
        elif "cross_scale_paragraphs" in data:
            # Structure like claude3.5_paragraphs.json
            source_data = data["cross_scale_paragraphs"]
        elif "cross_scale_connections" in data:
            # Structure like claude3.7_paragraphs.json
            source_data = data["cross_scale_connections"]
        elif "paragraphs" in data:
            # Structure like o3_paragraphs.json
            source_data = data["paragraphs"]
        else:
            print(f"Unknown structure in {file_path}")
            return
        
        # Process each item
        for item in source_data:
            standardized_item = {
                # Common fields that should be in all items
                "cross_scale_connection": item.get("cross_scale_connection", "No"),
                "feedback_loop_present": item.get("feedback_loop_present", "No"),
                "entities": item.get("entities", []),
                "relationship_type": item.get("relationship_type", "causal"),
                "loop_nature": item.get("loop_nature", "n/a"),
                "systemic_factor_involved": item.get("systemic_factor_involved", "No"),
                "scale_differences": item.get("scale_differences", {"spatial": False, "temporal": False}),
                "strength": item.get("strength", item.get("strength_rating", "3-Moderate")),
                "explicitness": item.get("explicitness", "Explicit"),
                "confidence": item.get("confidence", 0.8),
                "summary": item.get("summary", item.get("paragraph_text", item.get("paragraph", ""))),
                "pdf_source": item.get("pdf_source", item.get("source", "Unknown"))
            }
            
            # Convert strength rating if it's a number
            if isinstance(standardized_item["strength"], (int, float)):
                strength_map = {
                    1: "1-Very-Weak",
                    2: "2-Weak",
                    3: "3-Moderate",
                    4: "4-Above-Avg",
                    5: "5-Strong"
                }
                standardized_item["strength"] = strength_map.get(standardized_item["strength"], "3-Moderate")
            
            # Ensure scale_differences has the right structure
            if not isinstance(standardized_item["scale_differences"], dict):
                standardized_item["scale_differences"] = {"spatial": False, "temporal": False}
            
            standardized_data.append(standardized_item)
        
        # Generate the output file path
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_directory, file_name)
        
        # Write the standardized data to the new file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(standardized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Standardized {file_path} -> {output_file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all JSON files
for file_path in json_files:
    print(f"Processing {file_path}...")
    standardize_json(file_path)

print("All JSON files have been standardized.")
