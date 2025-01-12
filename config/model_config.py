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
        }
    }