from django.contrib import admin
from django.urls import path
from chatbot.views import home
from chatbot.views import sentiment_analysis

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('sentiment/', sentiment_analysis, name='sentiment_analysis'),
]