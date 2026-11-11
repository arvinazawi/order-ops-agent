import os
import json
import logging

import azure.functions as func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import OpenAI

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Locally this is your az login; in Azure it becomes the managed identity

credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

openai_client = OpenAI(
    base_url=os.environ["OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
    api_key=token_provider,
)

search_client = SearchClient(
    endpoint=os.environ["SEARCH_ENDPOINT"],
    index_name=os.environ["SEARCH_INDEX"],
    credential=credential,
)

CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]


#This is the grounding instruction, the part that makes it
# RAG rather than the model guessing from its training.

SYSTEM_PROMPT = """You are an order operations assistant for Contoso support staff.
Answer ONLY using the policy text provided in the context.
If the context does not contain the answer, say you don't know and suggest escalating.
Always cite the policy section number (for example: Section 1.2)."""


# search_text plus vector_queries together is hybrid search,
# keyword and meaning at once. VectorizableTextQuery lets Search embed the question itself,
# using the vectorizer the wizard set up, so our code never calls embed.

def retrieve_policy(question: str) -> str:
    vector_query = VectorizableTextQuery(
        text=question, k_nearest_neighbors=3, fields="text_vector"
    )
    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["chunk"],
        top=3,
    )
    chunks = [doc["chunk"] for doc in results]
    return "\n\n---\n\n".join(chunks)



#the HTTP endpoint (augment and generate).

@app.route(route="chat", methods=["POST"])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse('Send JSON like {"question": "..."}', status_code=400)

    question = body.get("question", "").strip()
    if not question:
        return func.HttpResponse("The 'question' field is required.", status_code=400)

    context = retrieve_policy(question)
    logging.info("Retrieved %d characters of context", len(context))

    response = openai_client.responses.create(
        model=CHAT_DEPLOYMENT,
        instructions=SYSTEM_PROMPT,
        input=f"Context:\n{context}\n\nQuestion: {question}",
    )

    return func.HttpResponse(
        json.dumps({"answer": response.output_text}),
        mimetype="application/json",
    )