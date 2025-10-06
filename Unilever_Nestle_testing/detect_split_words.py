import json
import re
import os
from collections import Counter

# Path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), 'all_nestle_unilever_paragraphs.json')

# Dictionary of common English word parts that might be split
# This helps identify when a word is likely split incorrectly
common_prefixes = ['un', 'in', 'im', 're', 'dis', 'en', 'em', 'non', 'de', 'ex', 'pre', 'pro', 'anti', 'auto', 'bi', 'co', 'con', 'com', 'counter', 'inter', 'intra', 'micro', 'mid', 'mis', 'over', 'semi', 'sub', 'super', 'trans', 'under', 'eco']
common_suffixes = ['able', 'ible', 'al', 'ial', 'ed', 'en', 'er', 'est', 'ful', 'ic', 'ing', 'ion', 'tion', 'ation', 'ity', 'ty', 'ive', 'ative', 'itive', 'less', 'ly', 'ment', 'ness', 'ous', 'ious', 'eous', 'ship', 'sion', 'tion', 'ate', 'ize', 'ise', 'ify', 'fy', 'ogy', 'al', 'ary', 'ery', 'ory', 'ize']

# Function to detect words with spaces inside them
def detect_split_words(text):
    split_words = []
    
    # This pattern looks for word parts separated by a space
    # where the parts could form a complete word when combined
    pattern = r'\b([a-zA-Z]{2,})\s+([a-zA-Z]{2,})\b'
    
    matches = re.finditer(pattern, text)
    for match in matches:
        first_part = match.group(1).lower()
        second_part = match.group(2).lower()
        full_word = first_part + second_part
        
        # Check if this looks like a split word based on common patterns
        is_likely_split = False
        
        # Check if the first part ends with a consonant and second part starts with a vowel (common split pattern)
        if first_part[-1] in 'bcdfghjklmnpqrstvwxyz' and second_part[0] in 'aeiou':
            is_likely_split = True
            
        # Check if the first part is a common prefix
        if first_part in common_prefixes:
            is_likely_split = True
            
        # Check if the second part is a common suffix
        if second_part in common_suffixes:
            is_likely_split = True
            
        # Check for specific known split words from our examples
        known_split_words = [
            'uncer tainty', 'con tribution', 'cancella tion', 'organ ised', 
            'struc ture', 'decen tralised', 'oppor tunity', 'bring ing', 
            'medi cine', 'estab lished', 'commu nication', 'discus sions', 
            'environ ment', 'entre preneurship', 'infor mation', 'in creased',
            'oppor tunities', 'pre paring', 'ex plores', 'com municable',
            'determin ation', 'eco nomic', 'fi nancial', 'op portunities'
        ]
        
        combined = f"{first_part} {second_part}"
        if combined in [w.lower() for w in known_split_words]:
            is_likely_split = True
            
        # If we think it's a split word, add it to our results
        if is_likely_split:
            original_match = match.group(0)
            split_words.append({
                'split_word': original_match,
                'combined': full_word,
                'context': text[max(0, match.start() - 30):min(len(text), match.end() + 30)]
            })
    
    return split_words

# Main function to process the JSON file
def main():
    try:
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
                found_words = detect_split_words(paragraph_text)
                
                # Update the dictionary with occurrences
                for word_info in found_words:
                    split_word = word_info['split_word']
                    if split_word in split_words_dict:
                        split_words_dict[split_word]['count'] += 1
                        split_words_dict[split_word]['sources'].add(pdf_source)
                    else:
                        split_words_dict[split_word] = {
                            'count': 1,
                            'sources': {pdf_source},
                            'combined': word_info['combined'],
                            'context': word_info['context']
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
            print(f"Context: \"...{info['context']}...\"")
        
        # Save results to a file
        with open('split_words_results.txt', 'w', encoding='utf-8') as out_file:
            out_file.write(f"Found {len(sorted_words)} cases of words split by spaces\n\n")
            for split_word, info in sorted_words:
                out_file.write(f"'{split_word}' → '{info['combined']}' (Found {info['count']} times)\n")
                out_file.write(f"Sources: {', '.join(list(info['sources'])[:3])}\n")
                out_file.write(f"Context: \"...{info['context']}...\"\n\n")
        
        print("\nResults saved to 'split_words_results.txt'")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
