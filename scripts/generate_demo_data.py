"""Generate fictitious insurance documents and deterministic claims data."""

import csv
import math
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.config import ROOT

DOCUMENTS = [
    ("avb_hausrat.pdf", "AVB Hausrat", "DEMO-AVB-HR-2026", [
        ("1. Zweck und Geltungsbereich", "Diese vollständig erfundenen Bedingungen gelten für die Muster-Hausratversicherung der fiktiven Demo Versicherung. Versicherungsort ist die in der Police bezeichnete Wohnung. Alle Beträge sind in CHF angegeben."),
        ("2. Versicherte Ereignisse", "Versichert sind Schäden am beweglichen Hausrat durch Feuer, Leitungswasser und Einbruchdiebstahl. Leitungswasserschäden umfassen bestimmungswidrig austretendes Wasser aus fest installierten Zu- und Ableitungen innerhalb der Wohnung."),
        ("3. Versicherungssumme und Selbstbehalt", "Die maximale Entschädigung je Ereignis entspricht der Versicherungssumme in der Police. Der reguläre Selbstbehalt beträgt CHF 200 je Ereignis. Für Einbruchdiebstahl gilt ein Selbstbehalt von CHF 300."),
        ("4. Ausschlüsse", "Nicht gedeckt sind normaler Verschleiss, vorsätzlich verursachte Schäden sowie Schäden durch Hochwasser von ausserhalb des Gebäudes. Einfacher Verlust ohne versichertes Ereignis ist ausgeschlossen."),
        ("5. Schadenmeldung", "Schäden sind innerhalb von fünf Kalendertagen nach Entdeckung zu melden. Benötigt werden die Policennummer, eine Beschreibung des Ereignisses, das Ereignisdatum, Fotos und verfügbare Kaufbelege. Bei Einbruchdiebstahl ist zusätzlich eine polizeiliche Bestätigung erforderlich."),
    ]),
    ("police_hausrat.pdf", "Police Hausrat", "DEMO-POL-0001", [
        ("1. Vertragsdaten", "Versicherungsnehmer: Musterperson A (synthetische Identität). Versicherungsort: Beispielweg 10, Musterstadt (fiktiv). Vertragsbeginn: 01.01.2026. Vertragsende: 31.12.2026. Policennummer: DEMO-POL-0001."),
        ("2. Versicherte Leistungen", "Versichert ist privater Hausrat gemäss DEMO-AVB-HR-2026. Die Versicherungssumme beträgt CHF 80 000 je Ereignis. Die vereinbarte Jahresprämie beträgt CHF 240. Es gelten die in den AVB genannten Selbstbehalte: CHF 200 regulär und CHF 300 bei Einbruchdiebstahl."),
        ("3. Besondere Vereinbarung", "Für Fahrräder innerhalb der versicherten Wohnung beträgt die Entschädigungsgrenze CHF 2 000 je Ereignis. Diebstahl ausserhalb der Wohnung ist durch diese Police nicht gedeckt. Weitere Zusatzdeckungen sind nicht vereinbart."),
        ("4. Verhältnis zu den AVB", "Diese Police und die AVB DEMO-AVB-HR-2026 bilden gemeinsam die synthetische Vertragsgrundlage. Bei ausdrücklich abweichenden besonderen Vereinbarungen geht die Police vor. Eine Auszahlung setzt die Prüfung des konkreten Schadenfalls voraus."),
    ]),
    ("avb_privathaftpflicht.pdf", "AVB Privathaftpflicht", "DEMO-AVB-PH-2026", [
        ("1. Versichertes Risiko", "Die fiktive Privathaftpflichtversicherung schützt die versicherte Privatperson bei berechtigten Schadenersatzansprüchen Dritter aus versehentlich verursachten Personen- und Sachschäden im privaten Alltag. Diese Bedingungen gehören nicht zur Hausratpolice DEMO-POL-0001."),
        ("2. Leistungen und Grenzen", "Die Deckungssumme beträgt CHF 5 000 000 je Ereignis. Bei Sachschäden beträgt der Selbstbehalt CHF 300. Bei Personenschäden entfällt der Selbstbehalt. Die Abwehr unberechtigter Ansprüche ist im Rahmen der Deckung enthalten."),
        ("3. Ausschlüsse", "Nicht versichert sind vorsätzlich verursachte Schäden, Schäden am eigenen Eigentum, berufliche Tätigkeiten sowie die Nutzung zulassungspflichtiger Motorfahrzeuge. Vertraglich freiwillig übernommene Haftungen über die gesetzliche Haftung hinaus sind ausgeschlossen."),
        ("4. Verhalten im Schadenfall", "Ein Ereignis ist innerhalb von fünf Kalendertagen nach Kenntnis zu melden. Die Meldung enthält die Beschreibung, das Datum und vorhandene Belege. Gegenüber Anspruchstellern darf ohne Abstimmung kein verbindliches Schuldanerkenntnis abgegeben werden."),
    ]),
]


def generate_claims(path: Path, count: int = 1200):
    rng = random.Random(42)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["customer_age", "claim_amount", "claim_type", "is_fraud"])
        for _ in range(count):
            age = rng.randint(18, 85)
            kind = rng.choice(["water", "fire", "theft", "liability"])
            amount = round(min(rng.lognormvariate(7.7, 1.0), 80000), 2)
            # Deliberately artificial signal plus stochastic noise, not actuarial evidence.
            logit = -3.6 + amount / 4500 + (1.6 if kind == "theft" else 0)
            fraud = int(rng.random() < 1 / (1 + math.exp(-logit)))
            writer.writerow([
                age if rng.random() > 0.03 else "",
                amount if rng.random() > 0.03 else "",
                kind if rng.random() > 0.02 else "", fraud,
            ])


def generate_pdf(path: Path, title: str, code: str, sections: list):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("DemoTitle", fontName="Helvetica-Bold", fontSize=25,
                              leading=30, textColor=colors.HexColor("#12314b"), spaceAfter=12))
    styles.add(ParagraphStyle("DemoBody", fontName="Helvetica", fontSize=10.5,
                              leading=15, alignment=TA_LEFT, spaceAfter=10))
    styles.add(ParagraphStyle("DemoSection", fontName="Helvetica-Bold", fontSize=12,
                              leading=16, textColor=colors.HexColor("#12314b"), spaceBefore=10, spaceAfter=5))
    story = [Paragraph("DEMO VERSICHERUNG / SYNTHETISCHE TESTDATEN", styles["Heading4"]),
             Spacer(1, 7 * mm), Paragraph(title, styles["DemoTitle"]),
             Paragraph(f"Dokument {code} | Stand 01.01.2026", styles["DemoBody"]),
             Paragraph("Frei erfundenes Beispieldokument für Softwaretests. Kein gültiger Versicherungsvertrag.", styles["DemoBody"])]
    for heading, body in sections:
        story.extend([Paragraph(heading, styles["DemoSection"]), Paragraph(body, styles["DemoBody"])])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d4dce3"))
        canvas.line(20 * mm, 18 * mm, 190 * mm, 18 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, 13 * mm, f"{code} | Nur zu Demonstrationszwecken")
        canvas.drawRightString(190 * mm, 13 * mm, f"Seite {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(path), pagesize=(210 * mm, 297 * mm),
                      rightMargin=20 * mm, leftMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=25 * mm,
                      title=title, author="Demo Versicherung (fiktiv)",
                      invariant=1).build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    generate_claims(data_dir / "claims.csv")
    for filename, title, code, sections in DOCUMENTS:
        generate_pdf(data_dir / filename, title, code, sections)
    print("Erzeugt: 1.200 synthetische Schäden und 3 Versicherungs-PDFs in data/.")


if __name__ == "__main__":
    main()
