import re
from pathlib import Path
import json

SCRIPT_DIR = Path.cwd()
base_dir = SCRIPT_DIR / "VeraAufgaben/"

pattern = r'^([A-Z])((:\s*)((.+)))*$'


for json_file in base_dir.rglob("*.json"):

    if "irrelevant" in str(json_file) or "borderline" in str(json_file) or "auditive_tasks" in str(json_file):
        continue
    
    # open and load json file
    with json_file.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

        # iterate through tasks
        for task in json_data['tasks']:
            # iterate through questions
            for question in task['questions']:
                
                # save answer possibilities in variable 
                answers = question['answers']
                gold_index = question["gold_answer"]
                # for each answer, check if it matches the pattern
                for idx, answer in enumerate(answers):

                    match = re.match(pattern, answer)
                    if match and idx == gold_index:
                        letter = match.group(1)
                        gold_answer = letter
                        question["gold_answer"] = gold_answer
#                        print(f"Processing file: {json_file}")
#                        print(f"Found answer option with letter '{letter}' and text '{text}'")
#                        print("---")
                    # else:
                    #     print(f"Processing file: {json_file}")
                    #     print(f"Answer option '{answer}' does not match the expected format.")
                    #     print("---")

    #Store result
    with json_file.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
 