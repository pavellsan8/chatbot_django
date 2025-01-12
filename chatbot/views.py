from django.shortcuts import render
from django.http import JsonResponse
from chatbot.model_files.roberta_gru.responding_model import SentimentResponseGenerator
from chatbot.model_files.distilbert_cnn.distilBERTCNN_model import ModelGPTIntegrator
from chatbot.model_files.distilbert.disitlBERT_model import ModelGPTIntegrator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from django.conf import settings
import os

def home(request):
    return render(request, 'chatbot.html')

# Responding Model
sentiment_generator = SentimentResponseGenerator()

# DstilBERT + CNN Model
# model_path = os.path.join(settings.BASE_DIR, 'chatbot', 'model_files', 'distilbert_cnn', 'distilbert_cnn_model.pth')
# sentiment_integrator = ModelGPTIntegrator(
#     model_path=model_path,
#     num_classes=7
# )

# DstilBERT + CNN Model
model_path = os.path.join(settings.BASE_DIR, 'chatbot', 'model_files', 'distilbert', 'distilbert_model.pth')
sentiment_integrator = ModelGPTIntegrator(
    model_path=model_path,
    num_classes=3
)

@csrf_exempt
@require_POST
def sentiment_analysis(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        if not text:
            return JsonResponse(
                {
                    'error': 'No text provided'
                }, status=400
            )
        
        # RoBERTa_GRU
        # result = sentiment_generator.process(text)

        # distilBERTCNN / distilBERT
        result = sentiment_integrator.analyze_text(text)
        return JsonResponse(result)
    
    except Exception as e:
        print("Exception occurred:", str(e))
        return JsonResponse(
            {
                'error': str(e)
            }, status=500
        )