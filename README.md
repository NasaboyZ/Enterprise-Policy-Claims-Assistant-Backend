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

### Alles mit Docker starten

Docker Desktop starten und im Backend-Verzeichnis ausführen:

```bash
docker compose up --build -d
```

Voraussetzung: Eine lokale `.env` mit `AION_API_KEY` (siehe `.env.example`).
Der Befehl baut das Image, bereitet ML-Modell und Dokumentenindex vor und startet
die API. Beim ersten Start wird das lokale Embedding-Modell heruntergeladen;
das kann einige Minuten dauern. Es wird kein Google-Key benötigt. Der Start
verbraucht kein Aion-Kontingent und führt keine Ragas-Evaluation aus.

- API und interaktive Dokumentation: **http://localhost:8000/docs**
- Erreichbarkeit: **http://localhost:8000/health**
- Startfortschritt und Logs: `docker compose logs -f backend`
- Status: `docker compose ps`
- Stoppen: `docker compose down`

Dokumente, Index, Modelle und Berichte bleiben in benannten Docker-Volumes
erhalten. `docker compose down -v` würde diese Daten löschen. Die Docker-Daten
sind getrennt von den lokal unter `data/`, `models/` und `reports/` gespeicherten
Dateien. Die API wird nur auf `127.0.0.1:8000` veröffentlicht; sie ist eine lokale
Demo ohne Benutzerverwaltung. Genau einen Worker verwenden und während Uploads
keinen zweiten Indexierungsprozess starten.

Evaluation bewusst separat starten (verbraucht zusätzliche Aion-Anfragen):

```bash
# Ein Testfall; liefert absichtlich Exitcode 2, weil der Gesamtbericht unvollständig ist.
docker compose exec backend python -m app.eval --limit 1

# Alle fünf Testfälle und vollständiges Quality Gate.
docker compose exec backend python -m app.eval
```

`GET /api/metrics` liefert den zuletzt gespeicherten Bericht. Vor der ersten
Auswertung antwortet der Endpoint mit HTTP 404. Ein API-Aufruf oder Upload
startet niemals automatisch eine Evaluation.

### Lokal ohne Docker

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

5. **API starten:**
   ```bash
   python -m app.ml_model
   python -m app.rag_engine index
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
   ```
   Die ersten beiden Befehle sind nur zur Vorbereitung bzw. Aktualisierung der
   lokalen Artefakte nötig. Die Endpoints sind dieselben wie im Container.

## REST-API

`POST /api/chat` verwendet den vorhandenen Agenten-Payload:

```bash
curl http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Welcher Selbstbehalt gilt bei Leitungswasser?","claim":{"customer_age":40,"claim_amount":100,"claim_type":"water"}}'
```

Die Antwort enthält `status`, `ml_score`, `final_answer`, `sources` und
`error_code`. `answered`, `manual_review` und `insufficient_context` sind normale
HTTP-200-Ergebnisse. Eingabefehler liefern 422, ausgeschöpfte API-Kontingente 429,
ungültige Modellantworten 502 und nicht verfügbare Dienste 503.

Genau eine neue PDF hochladen und sofort lokal indexieren:

```bash
curl http://localhost:8000/api/upload -F 'file=@/pfad/zur/police.pdf'
curl http://localhost:8000/api/metrics
```

Uploads dürfen höchstens 10 MiB und 100 Seiten enthalten. Verschlüsselte,
beschädigte und textlose PDFs werden abgewiesen; OCR ist nicht enthalten.
Neue Dateien erhalten einen Namen aus ihrem Inhalts-Hash. Identische Inhalte
werden erkannt, bestehende Dateien nicht überschrieben. Erfolgreiche neue
Uploads liefern HTTP 201, Duplikate HTTP 200. Bei Indexfehlern werden die neuen
Einträge zurückgenommen; andere Dokumente bleiben erhalten.

## Ragas-Evaluation und Quality Gate

```bash
python -m app.eval
python -m app.eval --faithfulness-threshold 0.80 --relevance-threshold 0.70
```

Die fünf Referenzfälle stehen in `data/eval_cases.json`. Der echte Agent läuft
mit einem festgelegten Niedrigrisiko-Schaden. Ragas 0.4.3 bewertet seine Antworten
gegen die tatsächlich abgerufenen Abschnitte. Aion dient ausdrücklich als
Antwortmodell und Bewerter; für die Relevanzberechnung werden lokale Embeddings
und eine Vergleichsfrage pro Antwort verwendet (`strictness=1`). Es gibt keine
zusätzlichen Prompt-Beispiele; der Bewerter verwendet `reasoning_effort=none`
und höchstens 2.048 Ausgabetokens, um das Kontingent zu schonen. Es gibt keine
automatischen Anbieterwechsel oder API-/Parser-Wiederholungen. Anfragen haben
mindestens fünf Sekunden Abstand. Eine vollständige Auswertung benötigt im
Normalfall etwa 20 Aion-Anfragen und kann das Tageskontingent überschreiten.

Berichte werden unter `reports/<run_id>.json` und atomar als `reports/latest.json`
gespeichert. Sie enthalten Konfiguration, Daten-Hashes, Referenzen, Antworten,
Kontexte, Einzelwerte, Mittelwerte und Fehlerstatus. Während eines Laufs keine
Dokumente hochladen oder den Index parallel aktualisieren.

Das Gate verlangt alle fünf beantworteten Fälle mit endlichen Messwerten und
Mittelwerten von mindestens 0.80 für Faithfulness und 0.70 für AnswerRelevancy.
Exitcodes: `0` bestanden, `1` Qualitätsgrenzen verfehlt, `2` unvollständig oder
technischer Fehler. Bei einem Kontingentfehler bleibt ein als unvollständig
markierter Bericht erhalten. Ein Teillauf ersetzt ebenfalls `latest.json`;
ältere Berichte bleiben unter ihrer Run-ID erhalten.

Diese Demo-Bewertung durch dasselbe Modell ist kein unabhängiger Nachweis
fachlicher Korrektheit oder Halluzinationsfreiheit. Referenzantworten dienen
auch der manuellen Nachprüfung. Offline-Tests simulieren die LLM-Antworten und
dürfen nicht mit einem erfolgreichen Live-Quality-Gate verwechselt werden.
