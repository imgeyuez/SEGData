"""
This script iterates through the downloaded source code from the IQB website and
extracts the filenames from the tasks to then match them to the tasks in the 
dataset and add on the key for which class the task is supposed to be.
"""

from bs4 import BeautifulSoup
from pathlib import Path
from collections import Counter, defaultdict
import os
import json
import re

# insert paths to source codes
paths = [
    r'Vera-SourceCodes\VERA-8 Mathematik - IQB Homepage.html',
    r'Vera-SourceCodes\VERA-8 Französisch - IQB Homepage.html',
    r'Vera-SourceCodes\VERA-8 Englisch - IQB Homepage.html',
    r'Vera-SourceCodes\VERA-8 Deutsch - IQB Homepage.html',
    r'Vera-SourceCodes\VERA-3 Mathematik - IQB Homepage.html',
    r'Vera-SourceCodes\VERA-3 Deutsch - IQB Homepage.html'
]

ROOT_DIR = Path(r"VeraDataset\VeraAufgaben")

regex_klasse = 'VERA-(\d+)'

aufgabe_klasse = {}

# Für Dublettenprüfung im HTML
html_name_counter = Counter()
html_name_sources = defaultdict(list)

for path in paths:
    match = re.search(r"VERA-(\d+)", path)
    if not match:
        print(f"Keine Klasse im Dateinamen gefunden: {path}")
        continue

    klasse = match.group(1)
    file_path = Path(path)

    with file_path.open(encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    titles = [h2.get_text(strip=True) for h2 in soup.select("h2.semi-bold.h3")]
    print(f"{file_path.name}: {len(titles)} gefundene Titel")

    for title in titles:
        html_name_counter[title] += 1
        html_name_sources[title].append(file_path.name)

        # Falls derselbe Aufgabenname in verschiedenen HTML-Dateien auftaucht:
        if title in aufgabe_klasse and aufgabe_klasse[title] != klasse:
            print(
                f"WARNUNG: Aufgabe '{title}' hat mehrere Klassen: "
                f"{aufgabe_klasse[title]} und {klasse}"
            )
        else:
            aufgabe_klasse[title] = klasse

# Dubletten im HTML melden
html_duplicates = {name: count for name, count in html_name_counter.items() if count > 1}

print("\n=== Doppelte Aufgaben im HTML ===")
if html_duplicates:
    for name, count in sorted(html_duplicates.items()):
        print(f"{name} ({count}x) in: {html_name_sources[name]}")
else:
    print("Keine doppelten Aufgaben im HTML gefunden.")


# JSON-Dateien durchgehen
json_name_counter = Counter()
json_name_to_paths = defaultdict(list)

updated_count = 0
missing_class = []
dataset_duplicates = []

for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        
        if not file.lower().endswith(".json"):
            continue

        json_path = Path(root) / file

        if "irrelevant" in str(json_path) or "borderline" in str(json_path):
            continue

        with json_path.open('r', encoding="utf-8") as f:
            json_f = json.load(f)

        aufgabenname = json_f['metadata']['title']

        json_name_counter[aufgabenname] += 1
        json_name_to_paths[aufgabenname].append(str(json_path))

        if aufgabenname in aufgabe_klasse:
            json_f.setdefault("metadata", {})
            json_f["metadata"]["class"] = aufgabe_klasse[aufgabenname]
            updated_count += 1

            # Datei wieder speichern
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(json_f, f, ensure_ascii=False, indent=2)
        else:
            missing_class.append({
                "aufgabenname": aufgabenname,
                "json_path": str(json_path)
            })


# Dubletten im Datensatz melden
dataset_duplicates = {
    name: paths for name, paths in json_name_to_paths.items() if len(paths) > 1
}

print("\n=== Doppelte Aufgaben im JSON-Datensatz ===")
if dataset_duplicates:
    for name, paths in sorted(dataset_duplicates.items()):
        print(f"{name} ({len(paths)}x)")
        for p in paths:
            print(f"  - {p}")
else:
    print("Keine doppelten Aufgaben im JSON-Datensatz gefunden.")


print("\n=== Aufgaben ohne neue Klasseninformation ===")
if missing_class:
    for item in missing_class:
        print(f"{item['aufgabenname']} -> {item['json_path']}")
else:
    print("Alle Aufgaben konnten einer Klasse zugeordnet werden.")

print(f"\nFertig. {updated_count} JSON-Dateien wurden aktualisiert.")