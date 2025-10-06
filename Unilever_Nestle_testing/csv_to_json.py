import csv
import json
import os
import glob
import ast

# Define the directory containing the CSV files
directory = os.path.dirname(os.path.abspath(__file__))

# Define the output directory for standardized JSON files
output_directory = os.path.join(directory, "standardized_json")

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, "*.csv"))

# Function to convert CSV to standardized JSON
def convert_csv_to_json(csv_file_path):
    try:
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        # Define the output JSON file path
        json_file_path = os.path.join(output_directory, f"{base_filename}.json")
        
        # Check if the JSON file already exists (we don't want to overwrite existing standardized files)
        if os.path.exists(json_file_path):
            print(f"Skipping {csv_file_path} as {json_file_path} already exists")
            return
        
        # Read the CSV file
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            # Create a CSV reader
            csv_reader = csv.DictReader(csvfile)
            
            # Convert each row to the standardized JSON format
            standardized_data = []
            for row in csv_reader:
                # Parse the entities field which is stored as a string representation of a list
                try:
                    entities = ast.literal_eval(row.get('entities', '[]'))
                except (SyntaxError, ValueError):
                    entities = []
                
                # Create the standardized item
                standardized_item = {
                    "cross_scale_connection": row.get('cross_scale_connection', 'No'),
                    "feedback_loop_present": row.get('feedback_loop_present', 'No'),
                    "entities": entities,
                    "relationship_type": row.get('relationship_type', 'causal'),
                    "loop_nature": row.get('loop_nature', 'n/a') if row.get('loop_nature') else 'n/a',
                    "systemic_factor_involved": "Yes" if row.get('systemic_factor') else "No",
                    "scale_differences": {
                        "spatial": True if row.get('cross_scale_connection') == 'Yes' else False,
                        "temporal": False  # Default to False as CSV doesn't specify this
                    },
                    "strength": convert_strength_rating(row.get('strength_rating')),
                    "explicitness": row.get('explicitness', 'Explicit'),
                    "confidence": float(row.get('confidence', 0.8)) if row.get('confidence') else 0.8,
                    "summary": row.get('paragraph_text', ''),
                    "pdf_source": row.get('pdf_source', 'Unknown')
                }
                
                standardized_data.append(standardized_item)
            
            # Write the standardized data to the JSON file
            with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(standardized_data, jsonfile, indent=2, ensure_ascii=False)
            
            print(f"Converted {csv_file_path} to {json_file_path}")
    
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")

# Function to convert strength rating to standardized format
def convert_strength_rating(strength_rating):
    if not strength_rating:
        return "3-Moderate"
    
    try:
        # If it's a number, convert to the standardized format
        rating = int(float(strength_rating))
        strength_map = {
            1: "1-Very-Weak",
            2: "2-Weak",
            3: "3-Moderate",
            4: "4-Above-Avg",
            5: "5-Strong"
        }
        return strength_map.get(rating, "3-Moderate")
    except (ValueError, TypeError):
        # If it's already a string in the right format, return it
        return strength_rating

# Process all CSV files
for csv_file_path in csv_files:
    print(f"Processing {csv_file_path}...")
    convert_csv_to_json(csv_file_path)

print("All CSV files have been converted to standardized JSON format.")
