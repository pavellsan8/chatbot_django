import torch
from transformers import DistilBertTokenizer
import numpy as np
from openai import OpenAI
import os
import json
from typing import List, Dict, Union, Optional
from django.conf import settings

import gc
import torch
from torch import nn
from transformers import DistilBertModel

class DistilBERT_CNN(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super(DistilBERT_CNN, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze some of the early BERT layers to prevent overfitting
        for param in self.bert.transformer.layer[:2].parameters():
            param.requires_grad = False
            
        # Multiple conv layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (k, 768), padding=(k//2, 0)) 
            for k in [3, 4, 5]
        ])
        
        self.layer_norm = nn.LayerNorm(32 * 3)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(32) for _ in range(3)
        ])
        
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flat = nn.Flatten()
        
        # Initialize fc layer here with the expected size
        # 32 * 3 because we have 3 conv layers with 32 filters each
        self.fc = nn.Linear(32 * 3, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent_id, mask):
        # Get BERT outputs
        self.bert.gradient_checkpointing_enable()
        outputs = self.bert(input_ids=sent_id, attention_mask=mask, return_dict=False)
        x = outputs[0].unsqueeze(1)

        # Apply multiple convolutions in parallel
        conv_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            conv_x = conv(x)
            conv_x = bn(conv_x)
            conv_x = self.relu(conv_x)
            conv_x = self.pool(conv_x)
            conv_outputs.append(conv_x)
        
        # Concatenate conv outputs
        x = torch.cat(conv_outputs, dim=1)
        x = self.flat(x)
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Use the pre-initialized fc layer
        x = self.fc(x)
        return self.softmax(x)

class ModelGPTIntegrator:
    def __init__(
        self, 
        model_path: str, 
        num_classes: int,
        sentiment_map: Optional[Dict[int, str]] = None
    ):

        # Set up OpenAI API key from Kaggle secrets
        # user_secrets = UserSecretsClient()
        # openai_key = user_secrets.get_secret("OpenAI API Key")
        
        # Initialize OpenAI client with the key from Kaggle secrets
        # if openai_api_key is None:
        openai_api_key = settings.OPENAI_API_KEY
        self.client = OpenAI(api_key=openai_api_key)
        
        # Set up sentiment mapping
        self.sentiment_map = sentiment_map or {
            0: "negative",
            1: "neutral or mixed emotion",
            2: "positive"
        }
        
        # Validate sentiment map
        if len(self.sentiment_map) != num_classes:
            raise ValueError(f"Sentiment map length must match number of classes ({num_classes})")
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, num_classes)
        self.model.eval()

    def _load_model(self, model_path: str, num_classes: int) -> torch.nn.Module:
        """Load the saved PyTorch model."""
        model = DistilBERT_CNN(num_classes=num_classes)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess input text for the DistilBERT model."""
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }

    def get_model_prediction(self, text: str) -> Dict[str, Dict[str, float]]:
        """Get prediction probabilities with sentiment labels."""
        # Preprocess and get prediction
        inputs = self.preprocess_text(text)
        with torch.no_grad():
            outputs = self.model(
                sent_id=inputs['input_ids'],
                mask=inputs['attention_mask']
            )
        
        # Convert to probabilities
        probs = outputs[0].cpu().numpy()
        
        # Create predictions dictionary
        numeric_probs = {f'class_{i}': float(prob) for i, prob in enumerate(probs)}
        sentiment_probs = {self.sentiment_map[i]: float(prob) for i, prob in enumerate(probs)}
        predicted_class = int(np.argmax(probs))
        
        return {
            'numeric_probabilities': numeric_probs,
            'sentiment_probabilities': sentiment_probs,
            'predicted_sentiment': self.sentiment_map[predicted_class],
            'confidence': float(np.max(probs))
        }

    def get_gpt4_analysis(
        self, 
        text: str, 
        predictions: Dict[str, Union[Dict[str, float], str, float]]
    ) -> Dict[str, str]:
        
        prompt = f"""
        Text: {text}
        
        Model Prediction: {predictions['predicted_sentiment']}
        Confidence: {predictions['confidence']:.2%}
        
        Based on this text and sentiment analysis, please provide:
        1. A brief confirmation or correction of the sentiment
        2. A response with the same sentiment as the user and encourages further dialogue
        
        Format your response exactly like this example:
        sentiment: neutral or mixed emotion
        response: I notice you have mixed feelings about this situation. Would you like to talk more about what's on your mind?
        
        Your response should have the same sentiment as the user. Make sure to maintain this exact format with the "sentiment:" and "response:" labels.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a conversation partner that will analyze the sentiment of the user and respond according to the sentiment. If it is negative, reply negatively. If it is positive, reply positively."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Parse the response to extract sentiment and response
        gpt_response = response.choices[0].message.content.strip()
        
        # Split the response into lines and parse
        lines = gpt_response.split('\n')
        sentiment_desc = lines[0].replace('sentiment:', '').strip()
        response_text = lines[1].replace('response:', '').strip()
        
        return {
            'sentiment_description': sentiment_desc,
            'response': response_text
        }

    def analyze_text(self, text: str) -> Dict[str, Union[str, int]]:
        
        predictions = self.get_model_prediction(text)
        gpt_analysis = self.get_gpt4_analysis(text, predictions)
        
        # Get the numeric sentiment value by finding the key in sentiment_map that matches the predicted sentiment
        sentiment_value = next(
            (k for k, v in self.sentiment_map.items() if v == predictions['predicted_sentiment']),
            3  # Default to neutral if not found
        )
        
        return {
            'input_text': text,
            'sentiment': sentiment_value,
            'sentiment_description': gpt_analysis['sentiment_description'],
            'response': gpt_analysis['response']
        }

# Example usage
if __name__ == "__main__":
    # Define custom sentiment mapping
    custom_sentiment_map = {
        0: "very negative",
        1: "neutral or mixed emotion",
        2: "very positive"
    }
    
    model_path = ''
    if model_path is None:
        model_path = os.path.join(
            settings.BASE_DIR, 
            'chatbot',
            'model_files',
            'distilbert_cnn',
            'distilbert_cnn_model.pth'
        )
    
    # Initialize integrator
    integrator = ModelGPTIntegrator(
        model_path=model_path,
        num_classes=3,
        sentiment_map=custom_sentiment_map
    )