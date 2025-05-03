import os
from flask import Flask, request, jsonify
import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool

app = Flask(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
# 检索服务指定语料库完整名称
CORPUS_NAME = os.environ.get("CORPUS_NAME")
GENERATIVE_MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-2.5-pro-preview-03-25")
# 设置返回数量以及相似度阈值
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", 15))
RETRIEVAL_DISTANCE_THRESHOLD = float(os.environ.get("RETRIEVAL_DISTANCE_THRESHOLD", 0.5))

# Initialize Vertex AI
if not all([PROJECT_ID, LOCATION, CORPUS_NAME]):
     raise ValueError("Missing required environment variables: GCP_PROJECT_ID, GCP_LOCATION, CORPUS_NAME")

vertexai.init(project=PROJECT_ID, location=LOCATION)

# 创建检索工具配置
rag_retrieval_config = rag.RagRetrievalConfig(
    top_k=RETRIEVAL_TOP_K,
    filter=rag.Filter(vector_distance_threshold=RETRIEVAL_DISTANCE_THRESHOLD),
)

rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=CORPUS_NAME,# 使用已经存在的语料库名称
                )
            ],
            rag_retrieval_config=rag_retrieval_config,
        ),
    )
)

try:
    # 使用检索工具加载 Generative Model
    rag_model = GenerativeModel(
        model_name=GENERATIVE_MODEL_NAME, tools=[rag_retrieval_tool]
    )
    print(f"Model '{GENERATIVE_MODEL_NAME}' loaded successfully with RAG tool for corpus '{CORPUS_NAME}'")
except Exception as e:
    print(f"Error loading model or setting up RAG tool: {e}")

@app.route("/", methods=["GET"])
def query():
    """
    Receives a query via HTTP GET request and returns a RAG response.
    Query should be provided as a URL parameter, e.g., /?query=什么是大数据？
    """
    user_query = request.args.get("query", default="")

    if not user_query:
        return jsonify({"error": "Please provide a 'query' parameter."}), 400

    print(f"Received query: {user_query}")

    try:
        response = rag_model.generate_content(user_query)
        response_text = response.text
        print(f"Generated response: {response_text[:100]}...") # Print beginning of response

        return jsonify({"query": user_query, "response": response_text})

    except Exception as e:
        print(f"Error during content generation: {e}")
        return jsonify({"error": f"An error occurred during the query: {e}"}), 500

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    print(f"Starting Flask app on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)