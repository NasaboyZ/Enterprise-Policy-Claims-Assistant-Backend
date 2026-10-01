# 🧠 Backend: Enterprise Policy & Claims Assistant

Dieses Repository enthält das Python-Backend für den **Enterprise Policy & Claims Assistant**. Es demonstriert den Aufbau einer modernen, hybriden KI-Architektur, die klassisches Machine Learning mit generativer KI (Large Language Models) und Agenten-Workflows verbindet.

Das Ziel dieses Projekts ist es, unstrukturierte Daten (Versicherungspolicen) und strukturierte Daten (Schadenshistorie) systematisch, sicher und halluzinationsfrei zu verarbeiten.

## Was in diesem Projekt konkret gezeigt wird

Dieses Backend deckt die Kernkompetenzen eines modernen AI Engineers ab:

### 1. RAG-Architektur & Vektordatenbanken

- **Document Ingestion:** Automatisches Einlesen von PDF-Policen.
- **Advanced Chunking:** Nutzung des `RecursiveCharacterTextSplitter` für semantisch sinnvolle Textabschnitte.
- **Vector Store:** Einsatz von **ChromaDB** und lokalen Deutsch/Englisch-Embeddings (`jinaai/jina-embeddings-v2-base-de`) über FastEmbed/ONNX für Similarity Search. Kein Embedding-API-Key nötig.

### 2. Klassisches Machine Learning (Statistik)

- **Fraud Detection:** Ein `Scikit-Learn` Modell (z. B. Random Forest) berechnet auf Basis strukturierter Kundendaten ein Betrugs- bzw. Risikoscore.
- **Hybrider Ansatz:** Das deterministische ML-Modell dient als Filter/Tool für das nicht-deterministische LLM.

### 3. Agent Frameworks (LangGraph)

- **Stateful Workflows:** Anstatt einfacher LLM-Ketten (Chains) wird **LangGraph** genutzt, um einen zyklischen, statusbasierten Agenten zu bauen.
- **Tool Calling:** Das LLM entscheidet dynamisch, wann es die Vektordatenbank (Retrieval) oder das ML-Modell (Risk Scoring) als Werkzeug aufrufen muss.

### 4. Systematische Qualitätsmessung (Evaluation)

- **Ragas Framework:** Die Antwortqualität des LLMs wird nicht dem Zufall überlassen. Das Backend enthält eine Evaluierungs-Pipeline, die Metriken wie **Faithfulness** (Halluzinations-Check) und **Answer Relevance** automatisiert misst.

### 5. API-Design & Deployment

- **FastAPI & Pydantic:** Bereitstellung einer sauberen, typsicheren REST-Schnittstelle.
- **Containerisierung:** Ein optimiertes `Dockerfile` stellt sicher, dass die Applikation Cloud-ready (z. B. für Azure) ist.

---

## Quickstart

1. **Abhängigkeiten installieren:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **API-Zugänge lokal konfigurieren:** Eine neue `.env` anhand von `.env.example`
   anlegen oder die bestehende Datei ergänzen. Vorhandene Schlüssel behalten.
   ```dotenv
   CHAT_PROVIDER=aion
   AION_API_KEY=YOUR_AION_KEY
   AION_CHAT_MODEL=aion-labs/aion-2.0
   ```
   AionLabs erzeugt standardmässig die Antworten. Indexierung und Suche berechnen
   Embeddings lokal auf der CPU; ein `GOOGLE_API_KEY` ist dafür nicht erforderlich.
   Schlüssel gehören ausschliesslich in die ignorierte `.env`; nur `.env.example`
   ohne Schlüssel wird versioniert. Um Antworten ausdrücklich über Gemini zu
   erzeugen, `CHAT_PROVIDER=gemini`, einen `GOOGLE_API_KEY` und optional
   `GEMINI_CHAT_MODEL=gemini-2.5-flash-lite` setzen. Bereits gesetzte
   Prozess-Umgebungsvariablen haben Vorrang vor `.env`.

   Aion verwendet `https://api.aionlabs.ai/v1/chat/completions`. Antworten werden
   lokal als JSON validiert und Quellenverweise geprüft; ungültige oder
   abgeschnittene Antworten werden verworfen. Es gibt keine automatischen
   Wiederholungen oder Wechsel zu einem anderen Anbieter. Eine Anfrage hat
   60 Sekunden Zeitlimit und maximal 4.096 Ausgabetokens einschliesslich Reasoning.

   Der [Aion-Free-Tarif](https://www.aionlabs.ai/docs/rate-limits/) nennt aktuell
   15 Anfragen pro Minute und 20.000 Tokens pro Tag (Stand: 1. Oktober 2026).
   Der tatsächliche Tarif wird im Aion-Konto verwaltet, nicht durch diese
   Konfiguration. Bei ausgeschöpftem Kontingent liefert das Backend eine
   verständliche Fehlermeldung. Das Modell ist auf Rollenspiel spezialisiert;
   die fachliche Antwortqualität muss mit den Beispielfragen geprüft werden.

3. **Dokumente lokal indexieren und suchen:**
   ```bash
   python -m app.rag_engine index
   python -m app.rag_engine search "Welcher Selbstbehalt gilt bei Leitungswasser?"
   ```
   Beim ersten Aufruf lädt FastEmbed das öffentliche
   [Deutsch/Englisch-Modell](https://huggingface.co/jinaai/jina-embeddings-v2-base-de)
   herunter (rund 640 MB Modelldaten). Dafür sind Internet und ausreichend
   Speicherplatz nötig, aber kein Konto oder API-Key. Danach wird der Cache unter
   `models/embeddings/` verwendet; die Embeddings laufen lokal ohne API-Gebühren.
   Dokumenttexte werden für die Embedding-Berechnung nicht hochgeladen.
   Der Modellcache wird von Git ignoriert.

   Der lokale Index verwendet die neue Collection `insurance_local_jina_de_v1`.
   Bestehende Gemini-/OpenAI-Collections bleiben erhalten und werden nicht mit
   lokalen Vektoren vermischt. Nach der Umstellung einmal `index` ausführen;
   weitere Aufrufe aktualisieren den Index und sind bei unveränderten PDFs idempotent.

4. **Offline-Tests ausführen:**
   ```bash
   python -m pytest
   ```
   Die Tests prüfen die API-Anbindung mit simulierten HTTP-Antworten und
   verbrauchen kein API-Kontingent. Sie belegen nicht die Qualität echter
   Modellantworten.
