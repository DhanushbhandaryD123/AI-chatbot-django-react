from django.urls import path
from .views import upload_document, chat # Ensure both are imported

urlpatterns = [
    path("upload/", upload_document, name="upload_document"),
    path("chat/", chat, name="chat"),
]