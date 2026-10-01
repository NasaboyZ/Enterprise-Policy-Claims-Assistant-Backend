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
- [x] Lokale Deutsch/Englisch-Embeddings (`jinaai/jina-embeddings-v2-base-de`) über FastEmbed konfigurieren; kein Google-/OpenAI-Key nötig.
- [x] ChromaDB als lokalen Vector Store aufsetzen und Chunks indizieren.
- [x] Retriever-Funktion mit Cosine Similarity (Top $k=3$) schreiben, die Text und Metadaten (Seitenzahlen) zurückgibt.

### Phase 4: LangGraph Agenten-Workflow (`app/agent.py`)

- [x] Agenten-Status (State) mittels `TypedDict` oder Pydantic definieren (Input-Query, Context, ML-Score, Final-Answer).
- [x] Tool 1: ML-Risiko-Check anbinden (lädt die `.pkl`-Datei und berechnet das Risiko für neue Eingaben).
- [x] Tool 2: ChromaDB Retriever anbinden.
- [x] Kontrollfluss-Knoten (Nodes) in LangGraph erstellen (Entscheidung: Ist der ML-Score hoch, geht es an den Sachbearbeiter; ist er niedrig, generiert das LLM die Antwort).
- [x] LLM-Generator-Knoten implementieren (inklusive striktem System-Prompt zur Quellen-Zitation).
- [x] Groq als Standard: `GROQ_API_KEY`, `openai/gpt-oss-120b`, striktes JSON-Schema und sichere Fehlermeldungen ohne Anbieterwechsel.
- [ ] Live-Antwort mit Groq prüfen, sobald der lokale `GROQ_API_KEY` eingetragen ist.

### Prüfstand für Phasen 1–4

- [x] 155 automatisierte Tests bestanden, inklusive Agent, lokaler Embedding-Anbindung, Ragas, REST-API und Upload-Rücknahme.
- [x] Abhängigkeiten geprüft (`pip check`) und alle im Plan vorgesehenen Bibliotheken erfolgreich importiert.
- [x] Drei synthetische PDFs gerendert und visuell geprüft; 1.200 Schadensdatensätze erzeugt.
- [x] Modell exportiert: Accuracy 0,7958, F1 0,4615 auf 240 synthetischen Testfällen.
- [x] Hochrisiko-Aufruf mit echtem Modell ohne API-Key geprüft; Ergebnis `manual_review`.
- [x] Echte lokale Embeddings: neun Dokumentabschnitte indexiert und Suche ohne Google-Key geprüft.
- [x] Aion-Live-Antwort für einen synthetischen Niedrigrisiko-Fall erzeugt.

Die automatisierte Suite verwendet Test-Doubles für neuronale Inferenz und externe APIs.
Zusätzlich wurden Upload, Suche und Agent-Ablauf mit echten lokalen Embeddings und echtem
ML-Modell geprüft (Antwort-LLM dort simuliert). Aion-Antworten wurden separat live geprüft.
Die Weiterleitung an Sachbearbeiter ist ein lokaler Status (`manual_review`), kein Nachrichtenversand.

### Phase 5: KI-Evaluation (`app/eval.py`)

- [x] Ragas 0.4.3 mit Groq als Standard (Aion optional) und lokalen Embeddings konfigurieren.
- [x] Fünf deutsche Ground-Truth-Fragen und Referenzantworten in `data/eval_cases.json` erstellen.
- [x] `Faithfulness` und `AnswerRelevancy` implementieren und mit echten Ragas-Metriken sowie simuliertem Groq-Transport testen.
- [x] Atomare JSON-Berichte, Quality Gate und Exitcodes 0/1/2 implementieren; Teilberichte bestehen das Gate nicht.
- [x] Evaluation nur per `python -m app.eval` starten; keine Auswertung bei API-Start, Upload oder Metrics-Abfrage.
- [ ] Vollständiges Live-Quality-Gate mit fünf Fällen bestanden; massgeblich ist der jeweilige Bericht, kein simulierter Test.

### Phase 6: REST API & Deployment (`app/main.py`)

- [x] FastAPI-App mit App-Factory und vorhandenen Pydantic-Modellen implementieren.
- [x] `POST /api/chat`: Agentenantwort, Quellen und verständliche Fehlerstatus.
- [x] `POST /api/upload`: einzelne PDF bis 10 MiB/100 Seiten prüfen und lokal indexieren; Inhalts-Deduplizierung und Rücknahme bei Fehlern.
- [x] `GET /api/metrics`: gespeicherten Ragas-Bericht lesen, vor erster Auswertung HTTP 404.
- [x] `GET /health`: Erreichbarkeit ohne Modell- oder API-Aufruf prüfen.
- [x] Dockerfile mit Python 3.14-slim gebaut; Nicht-Root-Benutzer, ein Worker, persistente Daten und lokale Portfreigabe.
- [x] `docker compose up --build -d` bereitet lokale Modelle und Index automatisch vor und startet die API; Befehl in README dokumentiert.
- [x] Container-Health, Hochrisiko-Chat, Upload-Deduplizierung und echte lokale Suche geprüft; Neustart erhält Modellcache und Dokumentenindex.

Python 3.10 aus dem ursprünglichen Plan wurde ersetzt, da aktuelle Pandas-/Scikit-Learn-
Abhängigkeiten mindestens Python 3.11 verlangen. Docker verwendet die lokal getestete
Python-3.14-Version. Das Backend ist eine lokale Demo ohne Authentifizierung und wird nur
auf `127.0.0.1:8000` veröffentlicht.

---
