import os
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from PIL import Image

# ========== CONFIG ==========
BASE_DIR = "VeraDataset\VeraAufgaben"

# Diese Ordner (egal auf welcher Ebene) komplett ignorieren
EXCLUDE_FOLDERS = {"auditive_tasks", "irrelevant", "borderline"}

# Erwartete Kategorien (für feste Reihenfolge + falls mal was fehlt)
QUESTION_TYPE_ORDER = [
    "spatial and geometric reasoning",
    "quantitative reasoning",
    "graph and pattern analysis",
    "text and diagram understanding",
]
# ============================

# ---- Functions ----
def should_skip_path(path: str) -> bool:
    """Skip, wenn irgendein Teil des Pfads in EXCLUDE_FOLDERS ist."""
    parts = set(os.path.normpath(path).split(os.sep))
    return not parts.isdisjoint(EXCLUDE_FOLDERS)

def iter_json_files(base_dir: str):
    for root, _, files in os.walk(base_dir):
        if should_skip_path(root):
            continue
        for fn in files:
            if fn.lower().endswith(".json"):
                yield os.path.join(root, fn)

def iter_png_files(base_dir: str):
    for root, _, files in os.walk(base_dir):
        if should_skip_path(root):
            continue
        for fn in files:
            if fn.lower().endswith(".png") or fn.lower().endswith(".json"):
                yield os.path.join(root, fn)

def detect_language_from_path(path: str) -> str:
    """Detect language marker from folder/file path."""
    path_upper = os.path.normpath(path).upper()
    if "ENG" in path_upper:
        return "ENG"
    elif "FRZ" in path_upper:
        return "FRZ"
    else:
        return "DEU"

# ---- Aggregation ----
# ---- About the Tasks ----
t_count = 0
q_count = 0
qtype_counter = Counter()
added_mc_counter = Counter()
bad_json = 0

png_count = 0
json_count = 0

answer_count_counter = Counter()
missing_answers = []
nonlist_answers = []

diff_qtypes = list()
diff_gold = list()

doc_level = Counter()
task_level = Counter()
question_level = Counter()

class_qtype_counter = defaultdict(Counter)
missing_level = []

language_doc_counter = Counter()
language_task_counter = Counter()
language_question_counter = Counter()

for jp in iter_png_files(BASE_DIR):
    if jp.lower().endswith(".png"):
        png_count += 1
    elif jp.lower().endswith(".json"):
        json_count += 1

for jp in iter_json_files(BASE_DIR):
    try:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        bad_json += 1
        continue

    doc_level[data['metadata']['class']] += 1

    lang = detect_language_from_path(jp)

    language_doc_counter[lang] += 1

    # count tasks
    tasks = data.get("tasks")
    t_count += len(tasks)
    task_level[data['metadata']['class']] += len(tasks)

    language_task_counter[lang] += len(tasks)

    # count questions
    for task in tasks:
        questions = task.get("questions")
        q_count += len(questions)
        question_level[data['metadata']['class']] += len(questions)
        language_question_counter[lang] += len(questions)

        for q in questions:
            qtype = q.get("question_type", "unknown")

            klasse = data["metadata"].get("class")
            class_qtype_counter[klasse][qtype] += 1

            if klasse == "k.A.":
                missing_level.append(jp)

            if qtype not in QUESTION_TYPE_ORDER:
                diff_qtypes.append(jp)

            if not isinstance(q["gold_answer"], int):
             diff_gold.append(jp) 

            qtype_counter[qtype] += 1

            if "added_multiplechoice" not in q:
                raise ValueError(
                    f"Missing key 'added_multiplechoice' "
                    f"in question {q.get('id', 'UNKNOWN')} "
                    f"(file: {jp})"
                )

            added = q["added_multiplechoice"]
            
            # manchmal als "true"/"false" String gespeichert -> normalisieren:
            if isinstance(added, str):
                added = added.strip().lower() == "true"
            added_mc_counter[added] += 1

            # ---- Anzahl Antwortmöglichkeiten pro Frage ----
            answers = q.get("answers", None)

            if answers is None:
                missing_answers.append(jp)
            else:
                if not isinstance(answers, list):
                    nonlist_answers.append(jp)
                else:
                    n_answers = len(answers)
                    answer_count_counter[n_answers] += 1

# ---- Output 1 & 2 ----
print("===== BASIS =====")
print((f"JSON-files gesamt: {json_count}"))
print((f"PNG-files gesamt: {png_count}"))
print(f"T-Aufgaben gesamt: {t_count}")
print(f"Q-Fragen gesamt:  {q_count}")
print(f"Nicht lesbare JSONs (übersprungen): {bad_json}")

# ---- Output 3: question_type counts + percentages ----
print("\n===== QUESTION TYPES =====")
total_q = sum(qtype_counter.values()) or 1

# feste Reihenfolge + alles andere hinten dran
ordered_types = [t for t in QUESTION_TYPE_ORDER if t in qtype_counter]
other_types = sorted([t for t in qtype_counter.keys() if t not in set(QUESTION_TYPE_ORDER)])
final_types = ordered_types + other_types

for t in final_types:
    n = qtype_counter[t]
    print(f"{t}: {n} ({n/total_q:.2%})")

print("Not matching QTypes:")
for q in diff_qtypes:
    print(q)

# ---- Output 4: added_multiplechoice ----
print("\n===== ADDED_MULTIPLECHOICE =====")
total_mc = sum(added_mc_counter.values()) or 1
for k in [False, True]:
    n = added_mc_counter.get(k, 0)
    print(f"{k}: {n} ({n/total_mc:.2%})")


print("Not matching Goldanswers:")
for g in diff_gold:
    print(g)

print("\n===== ANZAHL ANTWORTOPTIONEN =====")
total_ansq = sum(answer_count_counter.values()) or 1

for k in sorted(answer_count_counter.keys()):
    n = answer_count_counter[k]
    print(f"{k} Optionen: {n} ({n/total_ansq:.2%})")

# Optional: alles >= 5 zusammenfassen
ge5 = sum(v for k, v in answer_count_counter.items() if k >= 5)
if ge5:
    print(f">=5 Optionen (aggregiert): {ge5} ({ge5/total_ansq:.2%})")

if missing_answers:
    print(f"\nWARN: Fragen ohne 'answers' (Beispiele, bis 10): {len(missing_answers)}")
    for p in missing_answers[:10]:
        print(p)

if nonlist_answers:
    print(f"\nWARN: 'answers' ist nicht list (Beispiele, bis 10): {len(nonlist_answers)}")
    for p in nonlist_answers[:10]:
        print(p)

print("\n===== AUFGABEN LEVEL =====")
for key, value in doc_level.items():
    print(f"Dokumente für Klasse {key}: {value}")

for key, value in task_level.items():
    print(f"Tasks für Klasse {key}: {value}")

for key, value in question_level.items():
    print(f"Questions für Klasse {key}: {value}")


print("\n===== QUESTION TYPES PRO KLASSE =====")
for klasse in sorted(class_qtype_counter.keys(), key=str):
    print(f"\nKlasse {klasse}:")
    total = sum(class_qtype_counter[klasse].values()) or 1
    for qtype in final_types:
        n = class_qtype_counter[klasse].get(qtype, 0)
        print(f"  {qtype}: {n} ({n/total:.2%})")


print("\n===== LANGUAGE DISTRIBUTION =====")

total_docs = sum(language_doc_counter.values()) or 1
total_tasks = sum(language_task_counter.values()) or 1
total_questions = sum(language_question_counter.values()) or 1

for lang in ["DEU", "ENG", "FRZ"]:
    docs = language_doc_counter.get(lang, 0)
    tasks = language_task_counter.get(lang, 0)
    questions = language_question_counter.get(lang, 0)

    print(
        f"{lang}: "
        f"{docs} documents ({docs/total_docs:.2%}), "
        f"{tasks} tasks ({tasks/total_tasks:.2%}), "
        f"{questions} questions ({questions/total_questions:.2%})"
    )
    
# ---- Charts ----

# 3) question_type chart (bar with % labels)
labels = final_types
labels_cap = ["Graph and Pattern Analysis", "Quantitative Reasoning", "Spatial and Geometric Reasoning", "Text and Diagram Understanding"]

values = [qtype_counter[t] for t in labels]

# combine labels + values
pairs = list(zip(labels_cap, values))

# combine + sort
pairs = sorted(zip(labels_cap, values), key=lambda x: x[1], reverse=True)
labels_sorted, values_sorted = zip(*pairs)

plt.figure()
plt.bar(labels_sorted, values_sorted, color="lightseagreen")
plt.xticks(rotation=25, ha="right")
plt.ylabel("Number of Questions")
plt.title("Distribution of Question Types")

# Prozentlabels über Balken
for i, v in enumerate(values_sorted):
    plt.text(i, v, f"{(v/total_q)*100:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# 4) added_multiplechoice chart (bar)
mc_labels = ["False", "True"]
mc_values = [added_mc_counter.get(False, 0), added_mc_counter.get(True, 0)]

plt.figure()
plt.bar(mc_labels, mc_values, color="lightseagreen")
plt.ylabel("Number of Questions")
plt.title("Overview of manually added Multiple-Choice Tasks")

for i, v in enumerate(mc_values):
    plt.text(i, v, f"{(v/total_mc)*100:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()


# 5) answer options count chart
# Optional: >=5 zusammenfassen, damit die Achse nicht ausufert
bucketed = Counter()
for k, v in answer_count_counter.items():
    if k >= 5:
        bucketed[5] += v   # 5 steht hier für ">=5"
    else:
        bucketed[k] += v

opt_labels = []
opt_values = []
for k in sorted(bucketed.keys()):
    if k == 5 and any(orig_k >= 5 for orig_k in answer_count_counter.keys()):
        opt_labels.append(">=5")
    else:
        opt_labels.append(str(k))
    opt_values.append(bucketed[k])

total_opt = sum(opt_values) or 1

plt.figure()
plt.bar(opt_labels, opt_values, color="lightseagreen")
plt.ylabel("Number of Questions")
plt.title("Distribution of Answer Options")

for i, v in enumerate(opt_values):
    plt.text(i, v, f"{(v/total_opt)*100:.1f}%", ha="center", va="bottom")

plt.tight_layout()
plt.show()

