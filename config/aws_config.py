# config/aws_config.py

def get_aws_config():
    """Get AWS configuration settings"""
    return {
        'region': 'us-west-2',
        's3_bucket': 'systems-thinking-analysis',
        'sagemaker_role': 'arn:aws:iam::ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole',
        'sagemaker_container': 'ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/systems-thinking:latest',
        'lambda_functions': {
            'preprocess': 'systems-thinking-preprocess',
            'classify': 'systems-thinking-classify',
            'feedback': 'systems-thinking-feedback'
        },
        'dynamodb_tables': {
            'feedback': 'systems_thinking_feedback',
            'results': 'systems_thinking_results'
        },
        'opensearch': {
            'host': 'https://search-systems-thinking.us-west-2.es.amazonaws.com',
            'index': 'systems_thinking_embeddings'
        }
    }