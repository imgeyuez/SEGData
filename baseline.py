from pathlib import Path
import json

SCRIPT_DIR = Path(__file__).resolve().parent
base_dir = SCRIPT_DIR / "VeraAufgaben/"


# Track results
total_questions = 0
correct_answers = 0

for json_file in base_dir.rglob("*.json"):

    #TODO: Remove this check later, when all files have been checked for missing images and gold answers that are not integers
    if "irrelevant" in str(json_file) or "borderline" in str(json_file) or "auditive_tasks" in str(json_file):
        continue

    with (json_file.open("r", encoding="utf-8") as f):
        json_data = json.load(f)

        for task in json_data['tasks']:
            task_description = task['description']
            task_id = task['id']
            task_image_path = json_file.parent / task['image']


            for question in task['questions']:
                total_questions += 1

                question_id = question['id']
                question_text = question['text']
                answer_options =question['answers']
                question_gs = question['gold_answer']

                # Baseline: Always choose A
                if question_gs == 'A':
                    correct_answers += 1


print(f"Total questions: {total_questions}")
print(f"Correct answers: {correct_answers}")
print(f"Baseline accuracy: {correct_answers / total_questions:.2%}")
print("----")