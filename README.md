# 🧠 Backend: Enterprise Policy & Claims Assistant

Dieses Repository enthält das Python-Backend für den **Enterprise Policy & Claims Assistant**. Es demonstriert den Aufbau einer modernen, hybriden KI-Architektur, die klassisches Machine Learning mit generativer KI (Large Language Models) und Agenten-Workflows verbindet.

Das Ziel dieses Projekts ist es, unstrukturierte Daten (Versicherungspolicen) und strukturierte Daten (Schadenshistorie) systematisch, sicher und halluzinationsfrei zu verarbeiten.

## Was in diesem Projekt konkret gezeigt wird

Dieses Backend deckt die Kernkompetenzen eines modernen AI Engineers ab:

### 1. RAG-Architektur & Vektordatenbanken

- **Document Ingestion:** Automatisches Einlesen von PDF-Policen.
- **Advanced Chunking:** Nutzung des `RecursiveCharacterTextSplitter` für semantisch sinnvolle Textabschnitte.
- **Vector Store:** Einsatz von **ChromaDB** und OpenAI Embeddings (`text-embedding-3-small`) für effiziente Similarity Search.

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
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
