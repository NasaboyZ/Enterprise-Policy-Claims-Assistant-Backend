## 🛠️ Backend Development Plan (Python & AI)

### Phase 1: Projekt-Setup & Datenbasis

- [x] `venv` initialisieren und aktivieren.
- [x] `requirements.txt` anlegen und Abhängigkeiten installieren (`fastapi`, `langchain`, `langgraph`, `chromadb`, `scikit-learn`, `ragas`, `uvicorn`, `pydantic`).
- [x] Ordnerstruktur aufsetzen (`app/`, `data/`).
- [x] 2–3 synthetische Versicherungs-PDFs (z. B. AVB, Policen) im Ordner `data/` ablegen.
- [x] `claims.csv` mit historischen Schadensdaten (Kundenalter, Schadenssumme, Typ, `is_fraud`-Label) erstellen.

### Phase 2: Klassisches Machine Learning (`app/ml_model.py`)

- [x] CSV-Daten mit Pandas einlesen und Vorverarbeitung (z. B. fehlende Werte füllen) durchführen.
- [x] Features (X) und Target-Labels (y) trennen, `train_test_split` durchführen.
- [x] `RandomForestClassifier` (oder `LogisticRegression`) mit Scikit-Learn trainieren.
- [x] Modell evaluieren (F1-Score, Accuracy berechnen).
- [x] Trainiertes Modell als `.pkl`-Datei mittels `joblib` in einen `models/`-Ordner exportieren.

### Phase 3: RAG-Engine & Vektordatenbank (`app/rag_engine.py`)

- [x] `PyPDFLoader` (LangChain) integrieren, um die PDFs aus dem `data/`-Ordner einzulesen.
- [x] `RecursiveCharacterTextSplitter` implementieren (Chunk-Size: 600, Overlap: 80).
- [x] OpenAI Embeddings (`text-embedding-3-small`) konfigurieren.
- [x] ChromaDB als lokalen Vector Store aufsetzen und Chunks indizieren.
- [x] Retriever-Funktion mit Cosine Similarity (Top $k=3$) schreiben, die Text und Metadaten (Seitenzahlen) zurückgibt.

### Phase 4: LangGraph Agenten-Workflow (`app/agent.py`)

- [x] Agenten-Status (State) mittels `TypedDict` oder Pydantic definieren (Input-Query, Context, ML-Score, Final-Answer).
- [x] Tool 1: ML-Risiko-Check anbinden (lädt die `.pkl`-Datei und berechnet das Risiko für neue Eingaben).
- [x] Tool 2: ChromaDB Retriever anbinden.
- [x] Kontrollfluss-Knoten (Nodes) in LangGraph erstellen (Entscheidung: Ist der ML-Score hoch, geht es an den Sachbearbeiter; ist er niedrig, generiert das LLM die Antwort).
- [x] LLM-Generator-Knoten implementieren (inklusive striktem System-Prompt zur Quellen-Zitation).

### Prüfstand für Phasen 1–4

- [x] 30 automatisierte Tests bestanden, inklusive lokalem Gesamtworkflow mit echtem Modell, PDFs, Chroma und LangGraph.
- [x] Abhängigkeiten geprüft (`pip check`) und alle im Plan vorgesehenen Bibliotheken erfolgreich importiert.
- [x] Drei synthetische PDFs gerendert und visuell geprüft; 1.200 Schadensdatensätze erzeugt.
- [x] Modell exportiert: Accuracy 0,7958, F1 0,4615 auf 240 synthetischen Testfällen.
- [x] Hochrisiko-Aufruf mit echtem Modell ohne API-Key geprüft; Ergebnis `manual_review`.
- [ ] OpenAI-Live-Prüfung: `OPENAI_API_KEY` lokal konfigurieren, `python -m app.rag_engine index` ausführen und einen Agent-Aufruf mit niedrigem Risikoscore prüfen.

Die Implementierung der Phasen 3 und 4 ist lokal getestet. Embeddings und LLM wurden in den
Tests durch Test-Doubles ersetzt; ein Index mit echten OpenAI-Embeddings ist mangels API-Key
noch nicht erstellt. Die Checkmarks oben kennzeichnen implementierte und lokal geprüfte
Funktionen, nicht eine abgeschlossene OpenAI-Live-Prüfung. Die Weiterleitung an Sachbearbeiter
ist ein lokaler Status (`manual_review`), kein Nachrichtenversand.

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
