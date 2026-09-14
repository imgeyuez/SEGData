import argparse
import torch
from PIL import Image
from visionModels import model_mapping
from vllm import SamplingParams
from pydantic import BaseModel, create_model
from enum import Enum
from vllm.sampling_params import StructuredOutputsParams
from pathlib import Path
import json

# Argument parser
parser = argparse.ArgumentParser(description="Run a specific LLM model.")
parser.add_argument("--model", type=str, default="QWEN2_5VL", required=False, help="Name of the model to use (LLAVA, LLAVANext, Qwen2VL)")
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", required=False, help="Name of the specific model")
parser.add_argument("--explanation", action="store_true", default=True, help="If set, the model is asked to provide an explanation alongside the answer")
args = parser.parse_args()

# Get the selected model class
runner_class = model_mapping.get(args.model.upper())

if runner_class is None:
    raise ValueError(f"Unsupported model: {args.model}")


if args.model_name is not None:
    runner = runner_class(model_name=args.model_name)

else:
    runner = runner_class()

temperature = 0
max_tokens = 1024



SCRIPT_DIR = Path(__file__).resolve().parent
base_dir = SCRIPT_DIR / "VeraAufgaben/"

# Track results
total_questions = 0
correct_answers = 0
failed_parses = 0
results = []

#print GPU information
print("GPU Information:")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")


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

            image = Image.open(task_image_path)

            for question in task['questions']:
                total_questions += 1

                question_id = question['id']
                question_text = question['text']
                answer_options =question['answers']
                question_gs = question['gold_answer']

                ##<Build JSON dynamically, based on the answer options>
                # Extract just the letter keys from answer options (e.g., "A", "B", "C", "D")
                answer_keys = [opt.split(":")[0].strip() for opt in answer_options]

                # Dynamically create the Enum
                ErgebnisEnum = Enum("ErgebnisEnum", {key: key for key in answer_keys}, type=str)

                # Dynamically create the Pydantic model — with or without explanation field
                if args.explanation:
                    Antwort = create_model(
                        "Antwort",
                        Erklaerung=(str, ...),
                        Ergebnis=(ErgebnisEnum, ...)
                    )
                else:
                    Antwort = create_model(
                        "Antwort",
                        Ergebnis=(ErgebnisEnum, ...)
                    )

                json_schema = Antwort.model_json_schema()

                guided_decoding_params = StructuredOutputsParams(json=json_schema)
                ##</Build JSON dynamically, based on the answer options>

                question = "Ich zeige dir gleich eine Mathematikaufgabe, welche aus allgemeiner Beschreibung, Frage und einem Bild besteht. Antworte auf die Frage mit dem korrekten Zeichen (A,B,C,D,E,F). " +"\nBeschreibung: " +task_description +"\nFrage: "+question_text +"\nAntwortmöglichkeiten: " + ", ".join(answer_options)
                print(json_file, task_id, question_id)
                print(question)
                llm, prompt, stop_token_ids = runner.query(question)
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                    structured_outputs=guided_decoding_params,
                )

                inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                output_text = outputs[0].outputs[0].text

                print(output_text)
                print(f"Gold answer: {question_gs}")

                # Parse the JSON output into the Antwort object
                try:
                    antwort = Antwort.model_validate_json(output_text)

                    # Check if correct
                    is_correct = antwort.Ergebnis == question_gs
                    if is_correct:
                        correct_answers += 1

                    # Store result
                    result = {
                        'file': json_file.name,
                        'task_id': task_id,
                        'question_id': question_id,
                        'predicted': antwort.Ergebnis,
                        'gold': question_gs,
                        'correct': is_correct,
                        'number_of_options': len(answer_options),
                        'explanation': antwort.Erklaerung if args.explanation else None
                    }
                    results.append(result)
                except Exception as e:
                    failed_parses += 1
                    print(f"Error parsing output: {e}")
                    print(f"Raw output: {output_text}")
                    results.append({
                        'file': json_file.name,
                        'task_id': task_id,
                        'question_id': question_id,
                        'predicted': None,
                        'gold': question_gs,
                        'correct': False,
                        'number_of_options': len(answer_options),
                        'error': str(e)
                    })
                print("----")

# Calculate and print accuracy
accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Total Questions: {total_questions}")
print(f"Correct Answers: {correct_answers}")
print(f"Incorrect Answers: {total_questions - correct_answers - failed_parses}")
print(f"Failed Parses: {failed_parses}")
print(f"Accuracy: {accuracy:.2f}%")
print("="*50)

#SAve result
explanation_suffix = "with_explanation" if args.explanation else "without_explanation"
results_file = SCRIPT_DIR / f"results_{args.model}_{args.model_name.replace('/', '_')}{explanation_suffix}.json"
with results_file.open("w", encoding="utf-8") as f:
    json.dump({
        'model': args.model,
        'model_name': args.model_name,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'failed_parses': failed_parses,
        'accuracy': accuracy,
        'results': results
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {results_file}")