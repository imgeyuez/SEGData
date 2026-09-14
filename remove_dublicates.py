import os
import unicodedata
import re

def normalize_filename(name: str) -> str:
    # Unicode normalisieren
    name = unicodedata.normalize("NFKD", name)

    # typische kaputte Zeichen ersetzen
    replacements = {
        "Ф": "o",
        "ø": "o",
        "ö": "o",
        "Ö": "o",
        "ß": "ss"
    }

    for bad, good in replacements.items():
        name = name.replace(bad, good)

    # alles lowercase
    name = name.lower()

    # mehrere Leerzeichen / Unterstriche vereinheitlichen
    name = re.sub(r"[_\s]+", " ", name)

    return name


ROOT_DIR = "VeraAufgaben"

IGNORE_KEYWORDS = [
    "didaktische",
    "kommentierung",
    "auswertung",
    "losung",   # bewusst ohne Umlaut
    "loesung"
]

IGNORE_KEYWORDS = [normalize_filename(k) for k in IGNORE_KEYWORDS]


aufgaben = []

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        # nur PDFs
        if not file.lower().endswith(".pdf"):
            continue

        normalized_filename = normalize_filename(file)

        if any(keyword in normalized_filename for keyword in IGNORE_KEYWORDS):
            continue

        # get filename
        aufgabenname = os.path.splitext(file)[0]

        # relativer Pfad ab ROOT_DIR
        rel_path = os.path.relpath(root, ROOT_DIR)

        aufgaben.append((aufgabenname, rel_path))

# sortieren nach Aufgabenname, dann Pfad
aufgaben.sort(key=lambda x: (x[0].lower(), x[1].lower()))

print("Gefundene Aufgaben-Namen mit Pfad:")
for name, path in aufgaben:
    print(f"- {name}  →  {path}")

print(f"\nAnzahl gefundener PDFs: {len(aufgaben)}")
print(f"Anzahl eindeutiger Aufgaben: {len(set(n for n, _ in aufgaben))}")
