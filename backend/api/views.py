from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from chatbot.rag.loader import load_pdf
from chatbot.rag.chat import chat_with_pdf

@csrf_exempt
def upload_document(request):
    if request.method == "POST":
        file = request.FILES.get("file")
        path = f"uploads/documents/{file.name}"

        with open(path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        load_pdf(path)
        return JsonResponse({"message": "PDF uploaded and indexed"})

@csrf_exempt
def chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        question = data.get("question")

        try:
            answer = chat_with_pdf(question)
            return JsonResponse({"answer": answer})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
