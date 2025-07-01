import json
import re
import os
from collections import Counter
import enchant  # For spell checking

# Path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), 'all_nestle_unilever_paragraphs.json')

# Initialize English dictionary for spell checking
english_dict = enchant.Dict("en_US")

# List of known split words from our examples
known_split_patterns = [
    r'uncer\s+tainty', r'con\s+tribution', r'cancella\s+tion', r'organ\s+ised', 
    r'struc\s+ture', r'decen\s+tralised', r'oppor\s+tunit', r'bring\s+ing', 
    r'medi\s+cine', r'estab\s+lished', r'commu\s+nication', r'discus\s+sions', 
    r'environ\s+ment', r'entre\s+preneur', r'infor\s+mation', r'in\s+creased',
    r'pre\s+par', r'ex\s+plor', r'com\s+munic', r'determin\s+ation', 
    r'eco\s+nomic', r'fi\s+nancial', r'op\s+portunit'
]

# Function to detect words with spaces inside them
def detect_split_words(text, pdf_source, paragraph_index):
    split_words = []
    
    # First check for known patterns
    for pattern in known_split_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            split_word = match.group(0)
            combined_word = split_word.replace(" ", "")
            
            # Get context around the match
            start_pos = max(0, match.start() - 30)
            end_pos = min(len(text), match.end() + 30)
            context = text[start_pos:end_pos]
            
            split_words.append({
                'split_word': split_word,
                'combined': combined_word,
                'context': context,
                'pdf': pdf_source,
                'paragraph_index': paragraph_index
            })
    
    # Now look for other potential split words
    # Pattern to find two consecutive words where:
    # 1. First word is at least 2 letters
    # 2. Second word is at least 2 letters
    # 3. When combined, they form a valid English word
    pattern = r'\b([a-zA-Z]{2,})\s+([a-zA-Z]{2,})\b'
    
    for match in re.finditer(pattern, text):
        first_part = match.group(1)
        second_part = match.group(2)
        combined = first_part + second_part
        
        # Skip common phrases and short words
        if (first_part.lower() in ['the', 'and', 'but', 'for', 'nor', 'yet', 'so', 'as', 'at', 'by', 'to', 'in', 'of', 'on', 'or', 'up', 'is', 'it', 'be', 'we', 'us', 'he', 'me', 'my', 'our'] or
            second_part.lower() in ['the', 'and', 'but', 'for', 'nor', 'yet', 'so', 'as', 'at', 'by', 'to', 'in', 'of', 'on', 'or', 'up', 'is', 'it', 'be', 'we', 'us', 'he', 'me', 'my', 'our']):
            continue
        
        # Check if combined word exists in English dictionary
        if english_dict.check(combined.lower()):
            # Get context around the match
            start_pos = max(0, match.start() - 30)
            end_pos = min(len(text), match.end() + 30)
            context = text[start_pos:end_pos]
            
            split_words.append({
                'split_word': match.group(0),
                'combined': combined,
                'context': context,
                'pdf': pdf_source,
                'paragraph_index': paragraph_index
            })
    
    return split_words

# Main function to process the JSON file
def main():
    try:
        # Check if pyenchant is installed
        try:
            import enchant
        except ImportError:
            print("The 'pyenchant' package is required but not installed.")
            print("Please install it using: pip install pyenchant")
            return
            
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Loaded JSON file with {len(data)} paragraphs")
        
        # Dictionary to store split words and their occurrences
        split_words_dict = {}
        
        # Process each paragraph
        for i, entry in enumerate(data):
            if 'paragraph' in entry:
                paragraph_text = entry['paragraph']
                pdf_source = entry.get('pdf', 'Unknown PDF')
                
                # Find split words
                found_words = detect_split_words(paragraph_text, pdf_source, i)
                
                # Update the dictionary with occurrences
                for word_info in found_words:
                    split_word = word_info['split_word']
                    if split_word in split_words_dict:
                        split_words_dict[split_word]['count'] += 1
                        split_words_dict[split_word]['sources'].add(pdf_source)
                        split_words_dict[split_word]['occurrences'].append({
                            'pdf': pdf_source,
                            'paragraph_index': word_info['paragraph_index'],
                            'context': word_info['context']
                        })
                    else:
                        split_words_dict[split_word] = {
                            'count': 1,
                            'sources': {pdf_source},
                            'combined': word_info['combined'],
                            'occurrences': [{
                                'pdf': pdf_source,
                                'paragraph_index': word_info['paragraph_index'],
                                'context': word_info['context']
                            }]
                        }
        
        # Sort by frequency (most common first)
        sorted_words = sorted(split_words_dict.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Print results
        print("\n=== WORDS SPLIT BY SPACES ===")
        print(f"Found {len(sorted_words)} cases of words split by spaces")
        
        # Print the results
        for split_word, info in sorted_words:
            print(f"\n'{split_word}' → '{info['combined']}' (Found {info['count']} times)")
            print(f"Sources: {', '.join(list(info['sources'])[:3])}{'...' if len(info['sources']) > 3 else ''}")
            for i, occurrence in enumerate(info['occurrences'][:3]):  # Show up to 3 occurrences
                print(f"Occurrence {i+1}: \"{occurrence['context']}\"")
            if len(info['occurrences']) > 3:
                print(f"... and {len(info['occurrences']) - 3} more occurrences")
        
        # Save results to a file
        with open('extra_spaces_results.txt', 'w', encoding='utf-8') as out_file:
            out_file.write(f"Found {len(sorted_words)} cases of words split by spaces\n\n")
            for split_word, info in sorted_words:
                out_file.write(f"'{split_word}' → '{info['combined']}' (Found {info['count']} times)\n")
                out_file.write(f"Sources: {', '.join(list(info['sources'])[:3])}\n")
                for i, occurrence in enumerate(info['occurrences'][:3]):  # Show up to 3 occurrences
                    out_file.write(f"Occurrence {i+1}: \"{occurrence['context']}\"\n")
                if len(info['occurrences']) > 3:
                    out_file.write(f"... and {len(info['occurrences']) - 3} more occurrences\n")
                out_file.write("\n")
        
        print("\nResults saved to 'extra_spaces_results.txt'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
