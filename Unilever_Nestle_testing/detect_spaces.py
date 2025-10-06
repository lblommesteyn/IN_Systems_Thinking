import json
import re
import os

# Path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), 'all_nestle_unilever_paragraphs.json')

# Function to detect words with spaces inside them
def detect_words_with_spaces(text):
    # This regex looks for patterns where a word might be split by a space
    # It finds sequences where a letter is followed by a space and then another letter
    # within word boundaries or common punctuation
    words_with_spaces = []
    
    # Split the text into words
    words = text.split()
    
    # Check consecutive words to see if they might be parts of a single word
    for i in range(len(words) - 1):
        word1 = words[i].strip('.,;:!?()[]{}"\'-')
        word2 = words[i + 1].strip('.,;:!?()[]{}"\'-')
        
        # Check if these could be parts of a single word
        # Conditions: both parts are alphabetic, not common small words
        if (word1.isalpha() and word2.isalpha() and 
            len(word1) >= 2 and len(word2) >= 2 and
            word1.lower() not in ['the', 'and', 'but', 'for', 'nor', 'yet', 'so', 'as', 'at', 'by', 'to', 'in', 'of', 'on', 'or', 'up', 'is', 'it', 'be', 'we', 'us', 'he', 'me', 'my', 'our'] and
            word2.lower() not in ['the', 'and', 'but', 'for', 'nor', 'yet', 'so', 'as', 'at', 'by', 'to', 'in', 'of', 'on', 'or', 'up', 'is', 'it', 'be', 'we', 'us', 'he', 'me', 'my', 'our']):
            
            # Check if combining the words forms a valid English word
            # This is a simple heuristic - checking if the combined word "looks like" it could be a single word
            combined = word1.lower() + word2.lower()
            
            # Add to potential candidates
            words_with_spaces.append(f"{word1} {word2}")
    
    return words_with_spaces

# Main function to process the JSON file
def main():
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Loaded JSON file with {len(data)} paragraphs")
        
        # Dictionary to store words with spaces and their occurrences
        words_with_spaces_dict = {}
        
        # Process each paragraph
        for i, entry in enumerate(data):
            if 'paragraph' in entry:
                paragraph_text = entry['paragraph']
                pdf_source = entry.get('pdf', 'Unknown PDF')
                
                # Find potential words with spaces
                found_words = detect_words_with_spaces(paragraph_text)
                
                # Update the dictionary with occurrences
                for word in found_words:
                    if word in words_with_spaces_dict:
                        words_with_spaces_dict[word]['count'] += 1
                        words_with_spaces_dict[word]['sources'].add(pdf_source)
                    else:
                        words_with_spaces_dict[word] = {
                            'count': 1,
                            'sources': {pdf_source},
                            'example': paragraph_text
                        }
        
        # Sort by frequency (most common first)
        sorted_words = sorted(words_with_spaces_dict.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Print results
        print("\n=== WORDS POTENTIALLY SPLIT BY SPACES ===")
        print(f"Found {len(sorted_words)} potential cases of words split by spaces")
        
        # Print the top results
        for word, info in sorted_words[:50]:  # Show top 50
            print(f"\n{word} (Found {info['count']} times)")
            print(f"Sources: {', '.join(list(info['sources'])[:3])}{'...' if len(info['sources']) > 3 else ''}")
            print(f"Example: \"{info['example'][:100]}...\"")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
