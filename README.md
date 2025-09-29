
```
# Hotel Deduplication & URL Mismatch Checker

This project detects **duplicate hotels** and checks for **URL–name mismatches** in hotel datasets.  
It combines multiple similarity signals to increase accuracy:

- **Address similarity** → libpostal parsing + multilingual embeddings + house number heuristics  
- **Geographical similarity** → Haversine distance based scoring  
- **Name similarity** → BERT + TF-IDF + fuzzy matching  
- **URL mismatch** → compares `dealUrl` slugs vs. Trivago hotel names  

---

## 📂 Project Structure

All files remain in a single folder (no package/module imports required):

```


├── Dockerfile 

├── main.py

├── address_parsing.py

├── address_similarity.py

├── geo_similarity.py

├── bert_fuzzy_name_similarity.py

├── mismatch_checker.py

├── hotels_with_prices.xlsx   # sample input dataset

└── (output Excel files will be created here)


````

---

## 🐳 Run with Docker (recommended)

This project builds on top of **python:3.10-slim** and compiles **libpostal** from source.  
First ensure you have **Docker installed** (Linux, macOS, or Windows WSL2).

### 1. Build the image
```bash
docker build -t hotel-dedup .
````

### 2. Run the pipeline

Mount the current folder into the container so that outputs are saved locally:

```bash
docker run --rm -v $(pwd):/app hotel-dedup python main.py
```

On Windows PowerShell:

```powershell
docker run --rm -v ${PWD}:/app hotel-dedup python main.py
```

---

## 🧠 Models & Dependencies

The following models are downloaded automatically on first run:

* intfloat/multilingual-e5-large (address embeddings)
* paraphrase-multilingual-MiniLM-L12-v2 (name embeddings)

Key Python libraries (already installed via Dockerfile):

* pandas, numpy, openpyxl
* torch, sentence-transformers
* scikit-learn, rapidfuzz
* postal (libpostal bindings)
* tabulate

---

## 🚀 Pipeline Steps

1. Address parsing → normalize using libpostal
2. Address similarity → embeddings + numeric + weighted score
3. Geo similarity → Haversine distance → similarity score
4. Name similarity → hybrid BERT + TF-IDF + fuzzy score
5. Combine results → weighted average for duplicate candidates
6. URL mismatch check → Trivago name vs. dealer slug

---

## 📊 Outputs

After running main.py, the following Excel files will appear:

* final_similarity_candidates.xlsx → potential duplicate pairs
* url_hybrid_similarity.xlsx → URL–name similarity scores
* hybrid_mismatched_candidates.xlsx → flagged mismatches

---

## ✅ Example

```bash
docker run --rm -v $(pwd):/app hotel-dedup python main.py
```

Console output will show preview tables, and results will be saved as Excel files.

---

## ⚠️ Notes

* Running the pipeline the first time may take several minutes due to model downloads.
* If GPU is available inside Docker, PyTorch will use it automatically (otherwise CPU).
* The project is designed for experimentation; thresholds and weights can be tuned inside main.py or the respective modules.

```


