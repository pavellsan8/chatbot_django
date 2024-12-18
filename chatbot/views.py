from django.shortcuts import render
from django.http import JsonResponse
from chatbot.model_files.responding_model import SentimentResponseGenerator
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json

def home(request):
    return render(request, 'chatbot.html')

sentiment_generator = SentimentResponseGenerator()

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
        
        result = sentiment_generator.process(text)
        return JsonResponse(result)
    
    except Exception as e:
        return JsonResponse(
            {
                'error': str(e)
            }, status=500
        )