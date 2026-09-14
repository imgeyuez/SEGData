from pathlib import Path
import json
import matplotlib.pyplot as plt
import re

pattern = re.compile(r'^(?P<org>[^/]+)/(?P<model>.*?)-(?P<size>\d+[bB])(?:-|$)')

SCRIPT_DIR = Path(__file__).resolve().parent
base_dir = SCRIPT_DIR / "results/"

records = []
for json_file in base_dir.rglob("*.json"):
    print(json_file)

    with json_file.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

        record = {
            "model": json_data["model"],
            "model_name": json_data["model_name"],
            "total_questions": json_data["total_questions"],
            "correct_answers": json_data["correct_answers"],
            "failed_parses": json_data["failed_parses"],
            "accuracy": json_data["accuracy"]
        }

        match = pattern.search(record['model_name'])
        if match:
            record['org'] = match.group('org')
            record['model'] = match.group('model')
            record['size'] = match.group('size')
            record['label'] = record['model_name'].split('/')[-1]

        records.append(record)

# Sort by model name, then by size (numeric)
def sort_key(r):
    model = r.get('model', r['model_name'])
    size_str = r.get('size', '0b')
    size_num = int(re.sub(r'[bB]', '', size_str))
    return (model, size_num)

records.sort(key=sort_key)

# Print results
print("Model\tModel Name\tTotal Questions\tCorrect Answers\tFailed Parses\tAccuracy")
for record in records:
    print(f"{record['model']}\t{record['model_name']}\t{record['total_questions']}\t{record['correct_answers']}\t{record['failed_parses']}\t{record['accuracy']:.2%}")

# Plot results
models = [record['label'] for record in records]
accuracies = [record['accuracy'] for record in records]

sizes = sorted(set(int(re.sub(r'[bB]', '', r.get('size', '0b'))) for r in records))
colormap = plt.cm.viridis
size_to_color = {s: colormap(i / len(sizes)) for i, s in enumerate(sizes)}
colors = [size_to_color[int(re.sub(r'[bB]', '', r.get('size', '0b')))] for r in records]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=colors)
plt.xlabel('model_name')
plt.ylabel('Accuracy')
plt.title('Model Accuracy on Vera Aufgaben')
plt.ylim(0, 100)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "model_accuracies.png")
plt.show()