import os
from flask import Flask, request, jsonify
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

DELETE_EXISTING_CORPORA_STR = os.environ.get("DELETE_EXISTING_CORPORA", "false")
DELETE_EXISTING_CORPORA = DELETE_EXISTING_CORPORA_STR.lower() in ['true', 'yes', '1']

# 在应用上下文中初始化 Vertex AI
with app.app_context():
    if not PROJECT_ID:
        logger.error("PROJECT_ID environment variable not set!")
    else:
        logger.info(f"Initializing Vertex AI for project: {PROJECT_ID}, location: {LOCATION}")
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            logger.info("Vertex AI initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)

@app.route('/create_and_import', methods=['POST'])
def create_and_import_corpus():
    """
    列出现有语料库（删除），然后创建新语料库并导入文件
    """
    STRICT_DISPLAY_NAME = "xu_zuoyebang_rag_corpus"
    STRICT_EMBEDDING_MODEL = "publishers/google/models/text-multilingual-embedding-002"
    STRICT_GCS_PATHS = ["gs://xu-2025/computer_science/software_engineering/"]

    if not PROJECT_ID:
         return jsonify({"error": "GCP_PROJECT_ID environment variable not set, Vertex AI not initialized."}), 500

    corpus_name = None # 用于存储新创建的语料库名称

    try:
        logger.info("--- Starting RAG Corpus Setup and Import ---")

        logger.info("Listing existing corpora...")
        existed_corporas = rag.list_corpora()

        # 生产环境禁止删除语料库，仅做语料库切换
        if DELETE_EXISTING_CORPORA:
            logger.warning("DELETE_EXISTING_CORPORA is True. Attempting to delete all existing corpora.")
            for corpus in existed_corporas:
                try:
                    logger.info(f"Deleting corpus: {corpus.name}")
                    rag.delete_corpus(name=corpus.name)
                    logger.info(f"Corpus deleted: {corpus.name}")
                except Exception as e:
                    logger.error(f"Failed to delete corpus {corpus.name}: {e}", exc_info=True)

            # 删除后再次列出，确认是否都删除了
            existed_corporas_after_delete = rag.list_corpora()

        logger.info(f"Creating new corpus with display name: {STRICT_DISPLAY_NAME}")
        rag_corpus = rag.create_corpus(
            display_name=STRICT_DISPLAY_NAME,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                        publisher_model=STRICT_EMBEDDING_MODEL
                    )
                )
            ),
        )
        corpus_name = rag_corpus.name # 获取新创建的语料库名称
        logger.info(f"New corpus created successfully: {corpus_name}")

        logger.info("Listing corpora after creation...")
        logger.info(rag.list_corpora())

        logger.info(f"Starting file import into corpus: {corpus_name} from paths: {STRICT_GCS_PATHS}")

        import_operation_response = rag.import_files(
            corpus_name, # 使用新创建的语料库名称
            STRICT_GCS_PATHS,
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=512,
                    chunk_overlap=100,
                ),
            ),
            max_embedding_requests_per_min=1000,
        )

        logger.info(f"Import operation initiated successfully.")

        logger.info("--- RAG Corpus Setup and Import Process Initiated ---")

        # 返回一个成功的 JSON 响应
        return jsonify({
            "status": "Corpus setup and file import process initiated successfully",
            "corpus_name": corpus_name, # 返回新创建的语料库完整名称
            "message": "The import is running asynchronously. Check Cloud Logging or Vertex AI console for its final status and details."
        }), 200

    except Exception as e:
        # 捕获在整个流程中发生的任何异常
        logger.error(f"An error occurred during corpus setup or import initiation: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to initiate corpus setup or import process",
            "details": str(e)
            }), 500

@app.route('/', methods=['GET'])
def health_check():
    return "Indexing Service (Strict Logic) is running", 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
