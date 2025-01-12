import boto3
import logging
from typing import Dict
import json
from botocore.exceptions import ClientError

class AWSManager:
    def __init__(self, config: Dict):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.sagemaker_client = boto3.client('sagemaker')
        self.lambda_client = boto3.client('lambda')
        self.logger = logging.getLogger(__name__)
        
    def upload_to_s3(self, local_path: str, bucket: str, s3_key: str):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_path, bucket, s3_key)
        except ClientError as e:
            self.logger.error(f"Error uploading to S3: {str(e)}")
            raise
            
    def download_from_s3(self, bucket: str, s3_key: str, local_path: str):
        """Download file from S3"""
        try:
            self.s3_client.download_file(bucket, s3_key, local_path)
        except ClientError as e:
            self.logger.error(f"Error downloading from S3: {str(e)}")
            raise
            
    def deploy_sagemaker_endpoint(self, model_data: str, instance_type: str,
                                endpoint_name: str):
        """Deploy model to SageMaker endpoint"""
        try:
            model_response = self.sagemaker_client.create_model(
                ModelName=endpoint_name,
                PrimaryContainer={
                    'Image': self.config['sagemaker_container'],
                    'ModelDataUrl': model_data,
                },
                ExecutionRoleArn=self.config['sagemaker_role']
            )
            
            config_response = self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_name,
                ProductionVariants=[{
                    'VariantName': 'default',
                    'ModelName': endpoint_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': 1
                }]
            )
            
            endpoint_response = self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_name
            )
            
            return endpoint_response
            
        except ClientError as e:
            self.logger.error(f"Error deploying to SageMaker: {str(e)}")
            raise
            
    def invoke_lambda(self, function_name: str, payload: Dict) -> Dict:
        """Invoke Lambda function"""
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            return json.loads(response['Payload'].read())
        except ClientError as e:
            self.logger.error(f"Error invoking Lambda: {str(e)}")
            raise