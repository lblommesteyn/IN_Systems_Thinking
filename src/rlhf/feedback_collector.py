# src/rlhf/feedback_collector.py

from typing import Dict, List, Optional
import datetime
import boto3
from pydantic import BaseModel

class Feedback(BaseModel):
    text_id: str
    timestamp: datetime.datetime
    expert_id: str
    rating: int  # 1-5 scale
    dimension: Optional[str]
    comments: Optional[str]
    is_correct: bool
    suggested_corrections: Optional[Dict]

class FeedbackCollector:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.dynamo_client = boto3.client('dynamodb')
        self.table_name = config.get('feedback_table', 'systems_thinking_feedback')
        
    def store_feedback(self, feedback: Feedback):
        """Store feedback in DynamoDB"""
        item = {
            'text_id': {'S': feedback.text_id},
            'timestamp': {'S': feedback.timestamp.isoformat()},
            'expert_id': {'S': feedback.expert_id},
            'rating': {'N': str(feedback.rating)},
            'is_correct': {'BOOL': feedback.is_correct}
        }
        
        if feedback.dimension:
            item['dimension'] = {'S': feedback.dimension}
        if feedback.comments:
            item['comments'] = {'S': feedback.comments}
        if feedback.suggested_corrections:
            item['corrections'] = {'M': self._dict_to_dynamo(feedback.suggested_corrections)}
            
        self.dynamo_client.put_item(
            TableName=self.table_name,
            Item=item
        )
        
    def get_feedback_for_text(self, text_id: str) -> List[Feedback]:
        """Retrieve all feedback for a given text"""
        response = self.dynamo_client.query(
            TableName=self.table_name,
            KeyConditionExpression='text_id = :tid',
            ExpressionAttributeValues={
                ':tid': {'S': text_id}
            }
        )
        
        return [self._dynamo_to_feedback(item) for item in response['Items']]
        
    def _dict_to_dynamo(self, d: Dict) -> Dict:
        """Convert Python dict to DynamoDB format"""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = {'S': v}
            elif isinstance(v, bool):
                result[k] = {'BOOL': v}
            elif isinstance(v, (int, float)):
                result[k] = {'N': str(v)}
            elif isinstance(v, dict):
                result[k] = {'M': self._dict_to_dynamo(v)}
            elif isinstance(v, list):
                result[k] = {'L': [self._dict_to_dynamo({str(i): item})[str(i)] 
                                 for i, item in enumerate(v)]}
        return result
        
    def _dynamo_to_feedback(self, item: Dict) -> Feedback:
        """Convert DynamoDB item to Feedback object"""
        feedback_dict = {
            'text_id': item['text_id']['S'],
            'timestamp': datetime.datetime.fromisoformat(item['timestamp']['S']),
            'expert_id': item['expert_id']['S'],
            'rating': int(item['rating']['N']),
            'is_correct': item['is_correct']['BOOL'],
            'dimension': item.get('dimension', {}).get('S'),
            'comments': item.get('comments', {}).get('S'),
            'suggested_corrections': self._dynamo_to_dict(item.get('corrections', {}).get('M', {}))
        }
        return Feedback(**feedback_dict)
        
    def _dynamo_to_dict(self, dynamo_dict: Dict) -> Dict:
        """Convert DynamoDB format to Python dict"""
        result = {}
        for k, v in dynamo_dict.items():
            if 'S' in v:
                result[k] = v['S']
            elif 'N' in v:
                result[k] = float(v['N'])
            elif 'BOOL' in v:
                result[k] = v['BOOL']
            elif 'M' in v:
                result[k] = self._dynamo_to_dict(v['M'])
            elif 'L' in v:
                result[k] = [self._dynamo_to_dict({'item': item})['item'] 
                            for item in v['L']]
        return result

