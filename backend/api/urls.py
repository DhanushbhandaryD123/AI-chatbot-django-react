from django.urls import path
from .views import upload_document, chat

urlpatterns = [
    path('upload/', upload_document),
    path('chat/', chat),
]
