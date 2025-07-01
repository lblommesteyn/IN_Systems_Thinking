import json
import re
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import numpy as np

# Path to the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), 'all_nestle_unilever_paragraphs.json')
# Path for results
output_path = os.path.join(os.path.dirname(__file__), 'llm_detected_split_words.json')

# Initialize a lightweight model for text classification
# We'll use a small model suitable for sequence classification
model_name = "distilbert-base-uncased"  # A lightweight model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Function to detect potential split words using regex
def find_potential_split_words(text):
    # Pattern to find two consecutive words where both are at least 2 letters
    pattern = r'\b([a-zA-Z]{2,})\s+([a-zA-Z]{2,})\b'
    
    candidates = []
    for match in re.finditer(pattern, text):
        first_part = match.group(1)
        second_part = match.group(2)
        combined = first_part + second_part
        
        # Skip common phrases and short words
        common_words = ['the', 'and', 'but', 'for', 'nor', 'yet', 'so', 'as', 'at', 'by', 'to', 'in', 'of', 'on', 'or', 'up', 'is', 'it', 'be', 'we', 'us', 'he', 'me', 'my', 'our']
        if (first_part.lower() in common_words or second_part.lower() in common_words):
            continue
        
        # Get context around the match
        start_pos = max(0, match.start() - 50)
        end_pos = min(len(text), match.end() + 50)
        context = text[start_pos:end_pos]
        
        candidates.append({
            'split_word': match.group(0),
            'first_part': first_part,
            'second_part': second_part,
            'combined': combined,
            'context': context,
            'start': match.start(),
            'end': match.end()
        })
    
    return candidates

# Function to check if a word is likely split using the LLM
def check_with_llm(candidates, text):
    if not candidates:
        return []
    
    # Prepare prompts for the model
    prompts = []
    for candidate in candidates:
        # Create a prompt that asks if the word is likely split
        prompt = f"Context: '{candidate['context']}'\n\nIn this context, is '{candidate['split_word']}' likely a single word that was incorrectly split with a space? Should it be '{candidate['combined']}'? Answer yes or no."
        prompts.append(prompt)
    
    # Use the model to classify each candidate
    split_words = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 8
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_candidates = candidates[i:i+batch_size]
        
        # Tokenize the prompts
        inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Process the results
        for j, (candidate, prediction) in enumerate(zip(batch_candidates, predictions)):
            # If the model predicts "yes" with high confidence
            if prediction[1] > 0.7:  # Threshold for "yes"
                split_words.append({
                    'split_word': candidate['split_word'],
                    'combined': candidate['combined'],
                    'context': candidate['context'],
                    'confidence': float(prediction[1])
                })
    
    return split_words

# Alternative: Use a pre-trained text generation model for more accurate results
def setup_text_generation_pipeline():
    try:
        # Use a small text generation model
        return pipeline("text-generation", model="distilgpt2")
    except Exception as e:
        print(f"Error setting up text generation model: {e}")
        return None

def check_with_text_generation(candidates, text, pipeline):
    if not pipeline or not candidates:
        return []
    
    split_words = []
    
    for candidate in tqdm(candidates, desc="Checking candidates"):
        # Extract the context before the potential split word
        context_before = text[:candidate['start']].split()[-10:]  # Last 10 words before the split word
        context_before = " ".join(context_before) if context_before else ""
        
        # Generate text continuation with the combined word
        prompt_combined = f"{context_before} {candidate['combined']}"
        
        # Generate text continuation with the split word
        prompt_split = f"{context_before} {candidate['split_word']}"
        
        try:
            # Generate continuations for both versions
            result_combined = pipeline(prompt_combined, max_length=30, num_return_sequences=1)
            result_split = pipeline(prompt_split, max_length=30, num_return_sequences=1)
            
            # Compare the perplexity/likelihood of both continuations
            # Lower perplexity means the model finds that sequence more likely
            if result_combined[0]['score'] > result_split[0]['score']:
                split_words.append({
                    'split_word': candidate['split_word'],
                    'combined': candidate['combined'],
                    'context': candidate['context'],
                    'confidence': result_combined[0]['score'] - result_split[0]['score']
                })
        except Exception as e:
            print(f"Error processing candidate {candidate['split_word']}: {e}")
            continue
    
    return split_words

# Main function to process the JSON file
def main():
    try:
        # Load the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Loaded JSON file with {len(data)} paragraphs")
        
        # Initialize the text generation pipeline
        print("Setting up the language model...")
        text_gen_pipeline = setup_text_generation_pipeline()
        
        if not text_gen_pipeline:
            print("Warning: Could not set up text generation model. Using classification model instead.")
        
        # Dictionary to store split words and their occurrences
        all_split_words = []
        
        # Process each paragraph
        for i, entry in enumerate(tqdm(data, desc="Processing paragraphs")):
            if 'paragraph' in entry:
                paragraph_text = entry['paragraph']
                pdf_source = entry.get('pdf', 'Unknown PDF')
                
                # Find potential split words
                candidates = find_potential_split_words(paragraph_text)
                
                # Check candidates with the LLM
                if text_gen_pipeline:
                    split_words = check_with_text_generation(candidates, paragraph_text, text_gen_pipeline)
                else:
                    split_words = check_with_llm(candidates, paragraph_text)
                
                # Add source information
                for word in split_words:
                    word['pdf'] = pdf_source
                    word['paragraph_index'] = i
                    all_split_words.append(word)
                
                # Print progress
                if i % 10 == 0:
                    print(f"Processed {i} paragraphs, found {len(all_split_words)} split words so far")
        
        # Sort by confidence (highest first)
        all_split_words.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Save results to a JSON file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(all_split_words, out_file, indent=2, ensure_ascii=False)
        
        print(f"\nFound {len(all_split_words)} potential split words")
        print(f"Results saved to {output_path}")
        
        # Print top results
        print("\nTop 10 most confident split words:")
        for i, word in enumerate(all_split_words[:10]):
            print(f"{i+1}. '{word['split_word']}' → '{word['combined']}' (Confidence: {word.get('confidence', 'N/A'):.4f})")
            print(f"   Context: \"...{word['context']}...\"")
            print(f"   Source: {word['pdf']}")
            print()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
