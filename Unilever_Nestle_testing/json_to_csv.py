import json
import csv
import os
import glob

# Define the directory containing the standardized JSON files
json_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standardized_json")

# Define the output directory for CSV files
output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standardized_csv")

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get all JSON files in the directory
json_files = glob.glob(os.path.join(json_directory, "*.json"))

# Function to convert standardized JSON to CSV
def convert_json_to_csv(json_file_path):
    try:
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(json_file_path))[0]
        # Define the output CSV file path
        csv_file_path = os.path.join(output_directory, f"{base_filename}.csv")
        
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
        
        # Define CSV headers based on the standardized JSON structure
        fieldnames = [
            'cross_scale_connection',
            'feedback_loop_present',
            'entities',
            'relationship_type',
            'loop_nature',
            'systemic_factor_involved',
            'scale_differences_spatial',
            'scale_differences_temporal',
            'strength',
            'explicitness',
            'confidence',
            'summary',
            'pdf_source'
        ]
        
        # Write to CSV file
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            
            # Convert each JSON item to a CSV row
            for item in data:
                # Flatten the scale_differences dictionary
                scale_differences = item.get('scale_differences', {})
                
                # Create the CSV row
                # Clean the summary text - replace newlines with spaces
                summary = item.get('summary', '')
                if summary:
                    # Replace newlines with spaces
                    summary = summary.replace('\n', ' ').strip()
                
                row = {
                    'cross_scale_connection': item.get('cross_scale_connection', 'No'),
                    'feedback_loop_present': item.get('feedback_loop_present', 'No'),
                    'entities': str(item.get('entities', [])),
                    'relationship_type': item.get('relationship_type', 'causal'),
                    'loop_nature': item.get('loop_nature', 'n/a'),
                    'systemic_factor_involved': item.get('systemic_factor_involved', 'No'),
                    'scale_differences_spatial': scale_differences.get('spatial', False),
                    'scale_differences_temporal': scale_differences.get('temporal', False),
                    'strength': item.get('strength', '3-Moderate'),
                    'explicitness': item.get('explicitness', 'Explicit'),
                    'confidence': item.get('confidence', 0.8),
                    'summary': summary,
                    'pdf_source': item.get('pdf_source', 'Unknown')
                }
                
                writer.writerow(row)
        
        print(f"Converted {json_file_path} to {csv_file_path}")
    
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")

# Process all JSON files
for json_file_path in json_files:
    print(f"Processing {json_file_path}...")
    convert_json_to_csv(json_file_path)

print("All standardized JSON files have been converted to CSV format.")
