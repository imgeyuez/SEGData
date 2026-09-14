import argparse
import gc
import time
import psutil
from dotenv import load_dotenv

import torch
from PIL import Image
from visionModels import model_mapping
from vllm import SamplingParams


def log_ressources():
    ram = psutil.virtual_memory()
    print("RAM Consumtion:")
    print(f"  RAM used: {ram.used / 1024**3:.2f} GB")
    print(f"  RAM total: {ram.total / 1024**3:.2f} GB")
    print(f"  RAM percent: {ram.percent} %")

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    # Print details for each GPU
    for i in range(num_gpus):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")


def resize_image_max_side(image, max_side=450):
    original_width, original_height = image.size

    # Determine the scaling factor
    if original_width >= original_height:
        scale = max_side / original_width
    else:
        scale = max_side / original_height

    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize image with antialiasing
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def generate(runner, question=None, image=None, temperature=None, max_tokens=None):
    llm, prompt, stop_token_ids = runner.query(question)
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens, stop_token_ids=stop_token_ids
    )
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    return outputs[0].outputs[0].text


def format_model_results_to_markdown(results):
    # Step 1: Generate the summary table
    header = "| Model | Success |\n|--------|---------|"
    rows = []
    for model_name, data in results.items():
        rows.append(f'| {model_name} | {data.get("Success")} |')

    table = "\n".join([header] + rows)

    # Step 2: Generate the detailed section
    details = []
    for model_name, data in results.items():
        details.append(f"\n## {model_name}")
        error_msg = str(data.get("Error", "")).strip()
        output_msg = str(data.get("Output", "")).strip()
        details.append(f"**Error:** {error_msg}")
        details.append(f"**Output:** {output_msg}")

    return f"{table}\n\n" + "\n".join(details)


def clean_up(runner):
    wait = 10
    if runner:
        del runner
        time.sleep(wait)
        gc.collect()
        time.sleep(wait)
        torch.cuda.empty_cache()
        time.sleep(wait)


def main(models=None):

    load_dotenv()

    # Define parameters
    temperature = 0.2
    max_tokens = 128

    # Get image and prompt
    original_image = Image.open("example.jpg")
    image = resize_image_max_side(original_image)
    question = "Describe the following image:"

    # check if a predefined list of models was given
    if models:
        try:
            models_to_use = {model: model_mapping[model] for model in models}
        except KeyError:
            raise KeyError("Couldn't find a defined class for all models")
    else:
        models_to_use = model_mapping

    results = {}
    for name, model in models_to_use.items():
        results[name] = {}
        runner = None
        try:
            runner = model()
            output = generate(
                runner,
                question=question,
                image=image,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results[name]["Success"] = "yes"
            results[name]["Output"] = output

        except Exception as e:
            results[name]["Success"] = "no"
            results[name]["Error"] = e

        finally:
            print(f"\n{name}:\n")
            log_ressources()
            if runner:
                clean_up(runner)

        report = format_model_results_to_markdown(results)
        print(report)
        print("\n")
        with open("report.md", "w") as file:
            file.write(report)


def parse_arguments():

    parser = argparse.ArgumentParser(description="Test one or all vision-LLMs.")

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        type=str,
        default=[],
        help="Name of one model to test availability. Add multiple models divided by spaces.",
    )

    args = parser.parse_args()

    return args.models


if __name__ == "__main__":
    models = parse_arguments()
    main(models)
