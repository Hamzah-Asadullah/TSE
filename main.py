from sentence_transformers import SentenceTransformer
from os import path, makedirs
from numpy import argmax
from torch import Tensor
from http.server import BaseHTTPRequestHandler, HTTPServer
from json import loads, dumps

# TSE Text Similarity Endpoint
# Script written by Hamzah Asadullah
# Recommend 0.6B variant as it has only slighly degraded performance for 4x - 8x less computation

MODELS_PATH: str = "embedding-models"
MODEL_INDEX: int = 0
MODELS: list[str] = [ "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B" ]
MODEL_PATH = f"{MODELS_PATH}/{MODELS[MODEL_INDEX]}"
SERVER_PORT: int = 4998

LOG: bool = True
def log(text: str, ptype: str = 'n') -> None:
    if LOG:
        appendix: str = ""
        if ptype == 'n':
            appendix = "[NOTICE]: "
        elif ptype == 'u':
            appendix = "[UPDATE]: "
        elif ptype == 'w':
            appendix = "[WARNING]:"
        elif ptype == 'e':
            appendix = "[ERROR]:  "
        print(">>", appendix, text, end='\n' if ptype != 'u' else '\r')

def download_model(hf_path: str, save_to: str) -> None:
    tmp_model = SentenceTransformer(hf_path)
    tmp_model.save(save_to)

def get_similarity(model: SentenceTransformer, queries: list[str], documents: list[str], prompt: str) -> Tensor:
    query_embeddings: Tensor = model.encode(queries, prompt=prompt, convert_to_tensor=True)
    document_embeddings: Tensor = model.encode(documents, convert_to_tensor=True)
    return model.similarity(query_embeddings, document_embeddings)

def cutoff_str_list(input: list[str], max_length: int) -> list[str]:
    return [text[:max_length] for text in input]

log("Loading resources for server...", 'u')
makedirs(MODELS_PATH, exist_ok=True)
if not path.exists(MODEL_PATH):
    download_model(MODELS[MODEL_INDEX], MODEL_PATH)
model = SentenceTransformer(MODEL_PATH, device="cpu")

class SERVER(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_response_headers(self, http_code: int):
        self.send_response(http_code)
        self._set_headers()
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_headers()
        self.end_headers()
    
    def do_GET(self):
        if self.path == "/embedding-model":
            self._send_response_headers(200)
            self.wfile.write(bytes(MODELS[MODEL_INDEX], "utf-8"))
        else:
            self._send_response_headers(404)
        
    def do_POST(self):
        try:
            data: bytes = self.rfile.read(int(self.headers["Content-Length"]))
            data: dict[str, list[str] | str] = loads(data.decode("utf-8"))
        except Exception as e:
            log(f"Error trying to decode a request | {e}", 'e')
            self._send_response_headers(400)
            return

        if self.path == '/':
            try:
                queries: list[str] = cutoff_str_list(data["queries"], 8192 - 256)
                documents: list[str] = cutoff_str_list(data["documents"], 16 * 1024 - 256)
                prompt: str = data["prompt"][:150]
                similarity: list[float] = get_similarity(model, queries, documents, prompt).tolist()

                self.send_response(200)
                self._set_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(bytes(dumps(similarity), "utf-8"))
            except Exception as e:
                log(f"Error trying to compute response for a JSON-valid request", 'e')
                self._send_response_headers(400)
                return
        else:
            self._send_response_headers(404)
            return
    
def main():
    server = HTTPServer(("localhost", SERVER_PORT), SERVER)
    try:
        log(f"Server online on port {SERVER_PORT}, serving {MODELS[MODEL_INDEX]}.")
        server.serve_forever()
    except Exception as e:
        log(f"Got an except, exiting. | {e}")
        server.server_close()

if __name__ == "__main__":
    main()