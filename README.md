# TSE - Text Similarity Endpoint

**TSE (Text-Similarity-Endpoint)** is a lightweight Python 3.11 script that allows you to create a simple API endpoint for computing text similarities using Qwen3-series models. This makes it easy to compare texts or retrieve the most relevant document for a given query using embeddings.

---

## Features

- Lightweight and easy to set up.
- Uses Qwen3-series models for text embeddings.
- Provides a simple API endpoint for computing text similarity.
- Configurable model selection, download path, and server port.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Hamzah-Asadullah/TSE
cd TSE
````

2. Create a virtual environment:

```bash
python3.11 -m venv .venv
```

3. Activate the virtual environment:

* On Windows:

```bash
.\.venv\Scripts\activate
```

* On macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Start the server:

```bash
python main.py
```

> By default, it uses the 0.6B variant (\~3GB RAM, \~1.3GB download).

---

## Usage

Once the server is running, you can interact with it via a small helper function in Python:

```python
from requests import post

# Helper function
def get_similarity(queries: list[str], documents: list[str], prompt: str) -> list[list[float]]:
    payload = { "queries": queries, "documents": documents, "prompt": prompt }
    response = post("http://localhost:4998/", json=payload)
    return response.json()

# Example
questions = [
    "Who developed the world's first workable plastic magnet at room temperature?",
    "Who invented Algebra?"
]
answers = [
    "Hamzah Asadullah on GitHub",
    "Arthur Koestler",
    "Naveed Zaidi",
    "Sam Altman",
    "Muhammad ibn Musa al-Khwarizmi"
]
prompt = "Given a question, retrieve its correct 'inventor'"

similarity = get_similarity(questions, answers, prompt)
print(similarity)
```

### Interpreting the response

* Each sublist corresponds to a query, with similarity scores for each document.
* Example (using Qwen/Qwen3-Embedding-0.6B):

```text
[
  [0.459, 0.681, 0.569, 0.634, 0.602],
  [0.496, 0.622, 0.606, 0.577, 0.836]
]
```

* First query: highest value at index `[1]` → model thinks Arthur Koestler developed the plastic magnet (incorrect; correct: Naveed Zaidi).
* Second query: highest value at index `[4]` → model correctly identifies Muhammad ibn Musa al-Khwarizmi as the inventor of Algebra.

---

## Configuration

* **Model download path:** Modify `MODELS_PATH` on line 12 of `main.py`.
* **Model selection:** Change `MODEL_INDEX` on line 14.
* **Server port:** Modify `SERVER_PORT` on line 16.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes.
4. Submit a pull request.

---

## License

This project is open-source under the MIT License.
