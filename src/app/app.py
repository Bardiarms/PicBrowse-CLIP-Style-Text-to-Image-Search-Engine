from flask import Flask, render_template, request, send_from_directory
from src.app.retrieval_service import PicBrowseRetrievalService


app = Flask(__name__)

CHECKPOINT_PATH = "notebooks/local_service_test/artifacts/best_checkpoint.pt"               # Paths are for the local tests
VOCAB_PATH = "notebooks/local_service_test/artifacts/vocab.pt"
CACHED_EMBEDDINGS = "notebooks/local_service_test/artifacts/image_embeddings.pt"
IMAGE_FOLDER_PATH = "path/to/local/image/directory"

retrieval_service = PicBrowseRetrievalService(
                    checkpoint_path=CHECKPOINT_PATH,
                    vocab_path=VOCAB_PATH,
                    cached_image_embeddings_path=CACHED_EMBEDDINGS,
                    image_folder_path=IMAGE_FOLDER_PATH
)

@app.route("/", methods=["GET"])
def home():
   return render_template("index.html", results=None, query="")


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    
    if not query:
        return render_template("index.html", results=None, query="")
    
    results = retrieval_service.search_query(query=query, top_k=3)
    
    return render_template("index.html", results=results, query=query)
    
    
@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER_PATH, filename)