import json
import re
import os

# Path to the original JSON file
input_json_path = os.path.join(os.path.dirname(__file__), 'all_nestle_unilever_paragraphs.json')
# Path for the fixed JSON file
output_json_path = os.path.join(os.path.dirname(__file__), 'all_nestle_unilever_paragraphs_fixed.json')

# List of known split word patterns to fix
split_word_patterns = [
    # Format: (pattern, replacement)
    (r'uncer\s+tainty', 'uncertainty'),
    (r'con\s+tribution', 'contribution'),
    (r'cancella\s+tion', 'cancellation'),
    (r'organ\s+ised', 'organised'),
    (r'struc\s+ture', 'structure'),
    (r'decen\s+tralised', 'decentralised'),
    (r'oppor\s+tunit', 'opportunit'),  # Handles both opportunity and opportunities
    (r'bring\s+ing', 'bringing'),
    (r'medi\s+cine', 'medicine'),
    (r'estab\s+lish', 'establish'),  # Handles established, establishment
    (r'commu\s+nicat', 'communicat'),  # Handles communication, communicable
    (r'discus\s+sions', 'discussions'),
    (r'environ\s+ment', 'environment'),
    (r'envir\s+onment', 'environment'),
    (r'entre\s+preneur', 'entrepreneur'),
    (r'infor\s+mation', 'information'),
    (r'informa\s+tion', 'information'),
    (r'in\s+creased', 'increased'),
    (r'pre\s+par', 'prepar'),  # Handles preparing, preparation
    (r'ex\s+plor', 'explor'),  # Handles explore, exploring, explores
    (r'com\s+munic', 'communic'),
    (r'determin\s+ation', 'determination'),
    (r'eco\s+nomic', 'economic'),
    (r'fi\s+nancial', 'financial'),
    (r'op\s+portunit', 'opportunit'),
    (r'nutri\s+tion', 'nutrition'),
    (r'nutri\s+tional', 'nutritional'),
    (r'govern\s+mental', 'governmental'),
    (r'advertis\s+ing', 'advertising'),
    (r'effec\s+tive', 'effective'),
    (r'pro\s+vide', 'provide'),
    (r'par\s+ticularly', 'particularly'),
    (r'stud\s+ies', 'studies'),
    (r'look\s+ing', 'looking'),
    (r'recommen\s+dations', 'recommendations'),
    (r'ingre\s+dient', 'ingredient'),
    (r'con\s+tinue', 'continue'),
    (r'ro\s+bots', 'robots'),
    (r'bil\s+lion', 'billion'),
    (r'pos\s+sible', 'possible'),
    (r'Internation\s+al', 'International'),
    (r'aware\s+ness', 'awareness'),
    (r'includ\s+ing', 'including'),
    (r'deliver\s+ing', 'delivering'),
    (r'fresh\s+ly', 'freshly'),
    (r'de\s+veloping', 'developing'),
    (r'communica\s+tion', 'communication'),
    (r'engi\s+neer', 'engineer'),
    (r'im\s+proved', 'improved'),
    (r're\s+turned', 'returned'),
    (r'al\s+low', 'allow'),
    (r'expens\s+es', 'expenses'),
    (r'pre\s+mium', 'premium'),
    (r'sub\s+dued', 'subdued'),
    (r'confection\s+ery', 'confectionery'),
    (r're\s+turn', 'return'),
    (r'mar\s+ket', 'market'),
    (r'cat\s+egories', 'categories'),
    (r're\s+formu\s+lation', 'reformulation')
]

# Function to fix split words in text
def fix_split_words(text):
    fixed_text = text
    for pattern, replacement in split_word_patterns:
        fixed_text = re.sub(pattern, replacement, fixed_text, flags=re.IGNORECASE)
    return fixed_text

# Main function to process the JSON file
def main():
    try:
        # Load the original JSON file
        with open(input_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Loaded JSON file with {len(data)} paragraphs")
        
        # Process each paragraph to fix split words
        fixed_data = []
        for entry in data:
            if 'paragraph' in entry:
                # Create a copy of the entry
                fixed_entry = entry.copy()
                # Fix the paragraph text
                fixed_entry['paragraph'] = fix_split_words(entry['paragraph'])
                fixed_data.append(fixed_entry)
            else:
                # Keep entries without paragraphs unchanged
                fixed_data.append(entry)
        
        # Save the fixed JSON file
        with open(output_json_path, 'w', encoding='utf-8') as file:
            json.dump(fixed_data, file, indent=2, ensure_ascii=False)
        
        print(f"Fixed JSON saved to {output_json_path}")
        
        # Count how many paragraphs were modified
        modified_count = sum(1 for i, entry in enumerate(data) if 
                            'paragraph' in entry and 
                            'paragraph' in fixed_data[i] and 
                            entry['paragraph'] != fixed_data[i]['paragraph'])
        
        print(f"Modified {modified_count} paragraphs out of {len(data)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
