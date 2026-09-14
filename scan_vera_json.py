import sys
from pathlib import Path
import json

SCRIPT_DIR = Path.cwd()
base_dir = SCRIPT_DIR / "VeraAufgaben/"

nErrors = 0
for json_file in base_dir.rglob("*.json"):

    if "irrelevant" in str(json_file) or "borderline" in str(json_file) or "auditive_tasks" in str(json_file):
        continue

    with json_file.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

        if json_data['tasks'][0]['image'] is None:
            print(f"Processing file: {json_file}")
            continue

        image_path = json_file.parent / json_data['tasks'][0]['image']
        if not image_path.exists() or not image_path.is_file():

            #Skip path with subdirectory irrelevant and borderline
            if "irrelevant" in str(image_path) or "borderline" in str(image_path):
                continue

            nErrors+=1
            print(f"Working on file {json_file.name}")
            print(f"The path '{image_path}' does not exist")
            print("-----")

        for task in json_data['tasks']:
            task_description = task['description']
            task_id = task['id']
            if task['image'] is None:
                print(f"Processing file: {json_file}")
                continue

            task_image_path = json_file.parent / task['image']
            for question in task['questions']:
                question_id = question['id']
                question_text = question['text']
                question_gs = question['gold_answer']

                if not isinstance(question_gs, int):
                    print(f"Processing file: {json_file}")

                    print("It's not an integer!")


print(f"Found {nErrors}")