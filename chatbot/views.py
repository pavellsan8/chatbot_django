from django.shortcuts import render
from django.http import JsonResponse
# from chatbot.model_files.responding_model import SentimentResponseGenerator
from chatbot.model_files.distilBERTCNN_model import DistilBERT_CNN, ModelGPTIntegrator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from django.conf import settings
import os

def home(request):
    return render(request, 'chatbot.html')

# sentiment_generator = SentimentResponseGenerator()
# distil_bert_cnn = DistilBERT_CNN(7)
model_path = os.path.join(settings.BASE_DIR, 'chatbot', 'model_files', 'distilbert_cnn_model.pth')
sentiment_integrator = ModelGPTIntegrator(
    model_path=model_path,
    num_classes=7
)

@csrf_exempt
@require_POST
def sentiment_analysis(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        
        print(text)

        if not text:
            return JsonResponse(
                {
                    'error': 'No text provided'
                }, status=400
            )
        
        result = sentiment_integrator.analyze_text(text)
        return JsonResponse(result)
    
    except Exception as e:
        print("Exception occurred:", str(e))
        return JsonResponse(
            {
                'error': str(e)
            }, status=500
        )