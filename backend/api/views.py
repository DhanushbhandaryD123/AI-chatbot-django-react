import os
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from chatbot.rag.chat import chat_with_pdf 
from chatbot.rag import loader

BASE_DIR =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR =os.path.join(BASE_DIR, "uploads", "documents")

@csrf_exempt
@api_view(["POST"])
def upload_document(request):
    file=request.FILES.get("file")
    if not file:
        return Response({"message": "No file uploaded"}, status=400)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path=os.path.join(UPLOAD_DIR, file.name)

    with open(file_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

                    # Refresh the AI index after saving the file
    print(f"File {file.name} saved. Refreshing AI Index...")
    loader.vectorstore=loader.get_vectorstore() 

    if loader.vectorstore is None:
        return Response({
            "message": "File saved but AI indexing failed. Please verify your OpenAI API Key."
        }, status=500)
    return Response({"message": "File uploaded and indexed successfully"})

@csrf_exempt
@api_view(["POST"])
def chat(request):
    query=request.data.get("query")
    if not query:
        return Response({"answer": "Please enter a question."})
    try:
        answer=chat_with_pdf(query)
        return Response({"answer": answer})
    except Exception as e:
        return Response({"answer": f"AI Error: {str(e)}"}, status=500)