# src/utils/validation_utils.py

from typing import Dict, List, Optional, Tuple
import logging
from PIL import Image
import fitz  # PyMuPDF
import numpy as np

class ValidationTools:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def validate_pdf_mapping(self, pdf_path: str, extracted_paragraphs: List[Dict]) -> List[Dict]:
        """
        Validate the mapping between PDF and extracted paragraphs
        Returns list of validation results with coordinates and error flags
        """
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            results = []
            
            for paragraph in extracted_paragraphs:
                validation = {
                    'paragraph_id': paragraph['id'],
                    'text': paragraph['text'],
                    'page_num': paragraph['page_num'],
                    'coordinates': paragraph['coordinates'],
                    'errors': []
                }
                
                # Get PDF page
                page = doc[paragraph['page_num']]
                
                # Check if text exists at coordinates
                rect = fitz.Rect(paragraph['coordinates'])
                words = page.get_text("words", clip=rect)
                
                if not words:
                    validation['errors'].append('NO_TEXT_AT_LOCATION')
                    
                # Check text similarity
                extracted_text = " ".join([w[4] for w in words])
                if not self._text_similar(extracted_text, paragraph['text']):
                    validation['errors'].append('TEXT_MISMATCH')
                    
                # Validate section assignment
                if not self._validate_section(page, paragraph):
                    validation['errors'].append('SECTION_MISMATCH')
                    
                results.append(validation)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating PDF mapping: {str(e)}")
            raise
            
    def _text_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two text strings are similar using character-level comparison"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio() >= threshold
        
    def _validate_section(self, page: fitz.Page, paragraph: Dict) -> bool:
        """Validate if paragraph is assigned to correct section"""
        # Get text above paragraph to check for section headers
        above_rect = fitz.Rect(0, 0, page.rect.width, paragraph['coordinates'][1])
        above_text = page.get_text("text", clip=above_rect)
        
        # Check if assigned section appears in text above
        return paragraph['section'].lower() in above_text.lower()
        
    def visualize_mapping(self, pdf_path: str, extracted_paragraphs: List[Dict], 
                         output_path: str):
        """
        Create visual representation of paragraph mapping
        Highlights extracted paragraphs on PDF pages
        """
        doc = fitz.open(pdf_path)
        
        for paragraph in extracted_paragraphs:
            page = doc[paragraph['page_num']]
            rect = fitz.Rect(paragraph['coordinates'])
            
            # Add highlight
            highlight = page.add_highlight_annot(rect)
            
            # Add comment with paragraph ID
            highlight.set_info(content=f"ID: {paragraph['id']}")
            
        doc.save(output_path)
        
class ErrorAnalyzer:
    """Analyze and categorize extraction errors"""
    
    def __init__(self):
        self.error_categories = {
            'boundary': ['SPLIT_ERROR', 'MERGE_ERROR', 'MISSING_TEXT'],
            'section': ['WRONG_SECTION', 'MISSING_HIERARCHY'],
            'content': ['TEXT_CORRUPTION', 'ENCODING_ERROR', 'FORMAT_ERROR']
        }
        
    def analyze_errors(self, validation_results: List[Dict]) -> Dict:
        """
        Analyze validation results and provide error statistics
        Returns dict with error counts and patterns
        """
        stats = {
            'total_paragraphs': len(validation_results),
            'error_counts': {},
            'error_patterns': {},
            'page_distribution': {}
        }
        
        for result in validation_results:
            page_num = result['page_num']
            
            # Count errors by type
            for error in result['errors']:
                stats['error_counts'][error] = stats['error_counts'].get(error, 0) + 1
                
                # Track page distribution
                if error not in stats['page_distribution']:
                    stats['page_distribution'][error] = {}
                stats['page_distribution'][error][page_num] = \
                    stats['page_distribution'][error].get(page_num, 0) + 1
                    
            # Analyze error patterns
            if len(result['errors']) > 1:
                pattern = '+'.join(sorted(result['errors']))
                stats['error_patterns'][pattern] = \
                    stats['error_patterns'].get(pattern, 0) + 1
                    
        return stats
        
    def generate_error_report(self, stats: Dict) -> str:
        """Generate human-readable error report"""
        report = ["=== Error Analysis Report ===\n"]
        
        # Overall statistics
        report.append(f"Total paragraphs analyzed: {stats['total_counts']}")
        report.append(f"Paragraphs with errors: {sum(stats['error_counts'].values())}")
        report.append("")
        
        # Error counts by category
        report.append("Error Counts by Category:")
        for category, error_types in self.error_categories.items():
            category_count = sum(stats['error_counts'].get(et, 0) for et in error_types)
            report.append(f"{category.title()}: {category_count}")
            for error_type in error_types:
                if error_type in stats['error_counts']:
                    report.append(f"  - {error_type}: {stats['error_counts'][error_type]}")
        report.append("")
        
        # Error patterns
        report.append("Common Error Patterns:")
        sorted_patterns = sorted(stats['error_patterns'].items(), 
                               key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_patterns[:5]:
            report.append(f"  {pattern}: {count} occurrences")
            
        return "\n".join(report)

# config/model_config.py

def get_model_config():
    """Get model configuration settings"""
    return {
        'preprocessing': {
            'min_paragraph_length': 50,
            'max_paragraph_length': 2000,
            'min_sentences': 2,
            'max_sentences': 15,
            'excluded_sections': [
                'financial statements',
                'notes to financial statements',
                'independent auditor\'s report'
            ]
        },
        
        'embeddings': {
            'model_name': 'text-embedding-3-large',
            'dimension': 3072,
            'batch_size': 32
        },
        
        'classification': {
            'high_level': {
                'model_path': 'models/systems_thinking_classifier',
                'threshold': 0.75
            },
            'subdimension': {
                'model_path': 'models/subdimension_classifier',
                'threshold': 0.6
            }
        },
        
        'rag': {
            'retriever': {
                'similarity_threshold': 0.75,
                'top_k': 8
            },
            'context': {
                'max_tokens': 4096,
                'dedup_threshold': 0.8
            }
        },
        
        'rlhf': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'num_epochs': 3,
            'warmup_steps': 500
        },
        
        'validation': {
            'text_similarity_threshold': 0.8,
            'coordinate_tolerance': 5,  # pixels
            'section_validation': True
        }
    }

# requirements.txt

transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
boto3>=1.26.0
opensearch-py>=2.0.0
PyMuPDF>=1.22.0
Pillow>=10.0.0
pytesseract>=0.3.10
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0
pydantic>=2.0.0
tika>=2.6.0
PyPDF2>=3.0.0
scikit-learn>=1.3.0
python-dateutil>=2.8.2
logging>=0.5.0
pytest>=7.0.0
jupyterlab>=4.0.0