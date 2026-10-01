# Synthetische Beispieldaten

Alle Personen, Versicherer, Verträge, Beträge und Schäden sind erfunden. Keine echten Kundendaten.

| Datei | Inhalt |
| --- | --- |
| `avb_hausrat.pdf` | Hausrat-AVB mit Deckung, Ausschlüssen, Selbstbehalt und Meldeprozess |
| `police_hausrat.pdf` | Beispielpolice zur Hausrat-AVB mit Versicherungssumme und Sondervereinbarung |
| `avb_privathaftpflicht.pdf` | Separate Privathaftpflicht-AVB; keine Zusatzdeckung der Hausratpolice |
| `claims.csv` | 1.200 künstliche historische Schadenfälle, Seed 42 |

CSV-Schema:

| Spalte | Bedeutung |
| --- | --- |
| `customer_age` | Ganzzahliges Kundenalter, erzeugt zwischen 18 und 85 |
| `claim_amount` | Schadenssumme in CHF, zwei Dezimalstellen |
| `claim_type` | `water`, `fire`, `theft` oder `liability` |
| `is_fraud` | Künstliches binäres Trainingslabel: 1 = Betrugsfall, 0 = kein Betrugsfall |

Fehlende Featurewerte sind absichtlich enthalten (etwa 3 % bei Zahlen, 2 % beim Typ).
Labels fehlen nicht. Die Labelwahrscheinlichkeit hängt im Generator künstlich von Betrag und
Schadenstyp ab und enthält Zufallsrauschen; sie ist kein Versicherungs-Fachmodell.

`python -m scripts.generate_demo_data` erzeugt diese Dateien erneut.
`chroma/` wird erst durch die explizite Indexierung angelegt und nicht versioniert.
