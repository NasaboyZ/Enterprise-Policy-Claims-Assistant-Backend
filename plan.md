## 🛠️ Backend Development Plan (Python & AI)

### Phase 1: Projekt-Setup & Datenbasis

- [ ] `venv` initialisieren und aktivieren.
- [ ] `requirements.txt` anlegen und Abhängigkeiten installieren (`fastapi`, `langchain`, `langgraph`, `chromadb`, `scikit-learn`, `ragas`, `uvicorn`, `pydantic`).
- [ ] Ordnerstruktur aufsetzen (`app/`, `data/`).
- [ ] 2–3 synthetische Versicherungs-PDFs (z. B. AVB, Policen) im Ordner `data/` ablegen.
- [ ] `claims.csv` mit historischen Schadensdaten (Kundenalter, Schadenssumme, Typ, `is_fraud`-Label) erstellen.

### Phase 2: Klassisches Machine Learning (`app/ml_model.py`)

- [ ] CSV-Daten mit Pandas einlesen und Vorverarbeitung (z. B. fehlende Werte füllen) durchführen.
- [ ] Features (X) und Target-Labels (y) trennen, `train_test_split` durchführen.
- [ ] `RandomForestClassifier` (oder `LogisticRegression`) mit Scikit-Learn trainieren.
- [ ] Modell evaluieren (F1-Score, Accuracy berechnen).
- [ ] Trainiertes Modell als `.pkl`-Datei mittels `joblib` in einen `models/`-Ordner exportieren.

### Phase 3: RAG-Engine & Vektordatenbank (`app/rag_engine.py`)

- [ ] `PyPDFLoader` (LangChain) integrieren, um die PDFs aus dem `data/`-Ordner einzulesen.
- [ ] `RecursiveCharacterTextSplitter` implementieren (Chunk-Size: 600, Overlap: 80).
- [ ] OpenAI Embeddings (`text-embedding-3-small`) konfigurieren.
- [ ] ChromaDB als lokalen Vector Store aufsetzen und Chunks indizieren.
- [ ] Retriever-Funktion mit Cosine Similarity (Top $k=3$) schreiben, die Text und Metadaten (Seitenzahlen) zurückgibt.

### Phase 4: LangGraph Agenten-Workflow (`app/agent.py`)

- [ ] Agenten-Status (State) mittels `TypedDict` oder Pydantic definieren (Input-Query, Context, ML-Score, Final-Answer).
- [ ] Tool 1: ML-Risiko-Check anbinden (lädt die `.pkl`-Datei und berechnet das Risiko für neue Eingaben).
- [ ] Tool 2: ChromaDB Retriever anbinden.
- [ ] Kontrollfluss-Knoten (Nodes) in LangGraph erstellen (Entscheidung: Ist der ML-Score hoch, geht es an den Sachbearbeiter; ist er niedrig, generiert das LLM die Antwort).
- [ ] LLM-Generator-Knoten implementieren (inklusive striktem System-Prompt zur Quellen-Zitation).

### Phase 5: KI-Evaluation (`app/eval.py`)

- [ ] Ragas-Framework importieren und Setup konfigurieren.
- [ ] Test-Datensatz mit 5-10 Ground-Truth-Fragen und passenden Antworten erstellen.
- [ ] Metriken `Faithfulness` (Halluzinations-Check) und `AnswerRelevance` über die Test-Pipeline laufen lassen.
- [ ] Output als automatisierten JSON-Report (Quality Gate) speichern.

### Phase 6: REST API & Deployment (`app/main.py`)

- [ ] FastAPI-App initialisieren und Pydantic-Modelle für den API-Payload definieren.
- [ ] Endpoint `POST /api/chat` erstellen (nimmt Anfrage entgegen, triggert Agent, gibt Antwort + Quellen zurück).
- [ ] Endpoint `POST /api/upload` erstellen (für den Upload neuer PDFs, die direkt in ChromaDB indiziert werden).
- [ ] Endpoint `GET /api/metrics` erstellen (gibt die aktuellen Ragas-Scores zurück).
- [ ] `Dockerfile` (Python 3.10-slim) schreiben und das Backend-Image lokal bauen und testen.

---
