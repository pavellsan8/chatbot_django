import torch
from transformers import RobertaTokenizer, RobertaModel
from openai import OpenAI
from typing import Dict, Any
import os
from django.conf import settings

class RoBERTa_GRU(torch.nn.Module):
    def __init__(self):
        super(RoBERTa_GRU, self).__init__()
        self.RoBERTa = RobertaModel.from_pretrained('roberta-base')
        self.gru = torch.nn.GRU(input_size=768, hidden_size=256)
        self.flatten = torch.nn.Flatten()
        self.dense_1 = torch.nn.Linear(in_features=256, out_features=1000)
        self.gelu = torch.nn.GELU()
        self.dense_2 = torch.nn.Linear(in_features=1000, out_features=7)

    def forward(self, input_ids, attention_mask, token_type_ids):
        roberta_output = self.RoBERTa(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        cls = roberta_output.last_hidden_state[:, 0]
        sequences, _ = self.gru(cls.unsqueeze(0))
        flattened = self.flatten(sequences)
        x = self.dense_1(flattened)
        x = self.gelu(x)
        x = self.dense_2(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output

class SentimentResponseGenerator:
    def __init__(self, model_path: str = None, openai_api_key: str = None):
        if model_path is None:
            model_path = os.path.join(
                settings.BASE_DIR, 
                'chatbot',
                'model_files',
                'RoBERTa_GRU_model.pth'
            )
        
        if openai_api_key is None:
            openai_api_key = settings.OPENAI_API_KEY
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.sentiment_model = self.load_sentiment_model(model_path)
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()
        
        try:
            self.client = OpenAI(api_key=openai_api_key)
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None
    
    def load_sentiment_model(self, path: str):
        try:
            model = RoBERTa_GRU()
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                
                if isinstance(checkpoint, RoBERTa_GRU):
                    state_dict = checkpoint.state_dict()
                elif isinstance(checkpoint, dict):
                    state_dict = checkpoint
                else:
                    raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")
                    
                model.load_state_dict(state_dict)
            else:
                print(f"Warning: Model file not found at {path}")
            
            return model
        except Exception as e:
            print(f"Error loading sentiment model: {e}")
            return None
        
    def predict_sentiment(self, text: str) -> int:
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512,
                add_special_tokens=True
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.sentiment_model(
                    input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs.get('token_type_ids', torch.zeros_like(inputs['input_ids']))
                )
                predicted_class = torch.argmax(logits, dim=1).item()
            
            return predicted_class
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return 3  # Return neutral sentiment as fallback
    
    def generate_response(self, text: str, sentiment: int) -> str:
        if not self.client:
            return "ChatBot is currently unavailable. Please check your OpenAI API key."
            
        sentiment_map = {
            0: "very negative",
            1: "moderately negative",
            2: "slightly negative",
            3: "neutral or mixed emotion",
            4: "slightly positive",
            5: "moderately positive",
            6: "very positive"
        }
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using a standard model instead of fine-tuned
                messages=[
                    {
                        "role": "system", 
                        "content": f"You are an empathetic assistant responding to a {sentiment_map[sentiment]} sentiment message. Provide a helpful and appropriate response."
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def process(self, text: str) -> Dict[str, Any]:
        try:
            sentiment = self.predict_sentiment(text)
            response = self.generate_response(text, sentiment)
            
            sentiment_descriptions = {
                0: "very negative",
                1: "moderately negative",
                2: "slightly negative",
                3: "neutral or mixed emotion",
                4: "slightly positive",
                5: "moderately positive",
                6: "very positive"
            }
            
            return {
                "input_text": text,
                "sentiment": sentiment,
                "sentiment_description": sentiment_descriptions[sentiment],
                "response": response
            }
        except Exception as e:
            print(f"Error in process: {e}")
            return {
                "input_text": text,
                "sentiment": 3,
                "sentiment_description": "neutral or mixed emotion",
                "response": "I apologize, but I'm having trouble processing your message right now. Please try again."
            }

# Initialize the generator only once
sentiment_generator = SentimentResponseGenerator()