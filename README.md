# VeraDataset
Generation of a dataset consisting of vera-Aufgaben which includes various fields which can then be used to evaluate LLMs capabilities in various fields of tasks.

# Image conversion

## Convert EPS to PNG
```bash
convert -density 300 "$eps_file" -trim +repage  -background white -flatten "$png_file"
```

## Align multiple images
```bash
convert Geobrettfigurauslegen_Aufgabe_1.png Geobrettfigurauslegen_Aufgabe_2.png -background white -gravity center -append output.png
```


## Set up the conda project
```bash
conda create -n VERA_Dataset python=3.12 -vv --solver=classic -y
conda activate VERA_Dataset
pip install 
```



## Check out project
```bash
git clone https://github.com/imgeyuez/VeraDataset.git
cd VeraDataset
git checkout vllm
```

## Run model test
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python testModels.py
```

## Run Models



# QWEN2.5-VL
## QWEN/Qwen2.5-VL-3B-Instruct
```bash
srun --partition=${smallpartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py
```

## Qwen/Qwen2.5-VL-32B-Instruct
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=2 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model QWEN2_5VL --model-name Qwen/Qwen2.5-VL-32B-Instruct
```

## Qwen/Qwen2.5-VL-72B-Instruct-AWQ
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model QWEN2_5VL --model-name Qwen/Qwen2.5-VL-72B-Instruct-AWQ
```

```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=4 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model QWEN2_5VL --model-name Qwen/Qwen2.5-VL-72B-Instruct
```

# QWEN 3

## Qwen/Qwen3-VL-8B-Instruct
```bash
srun --partition=${smallpartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-8B-Instruct
```

## Qwen/Qwen3-VL-8B-Thinking
```bash
srun --partition=${smallpartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-8B-Thinking
```


## Qwen/Qwen3-VL-32B-Instruct-FP8
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=2 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-32B-Instruct-FP8
```

## Qwen/Qwen3-VL-32B-Thinking
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=2 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-32B-Thinking-FP8
```

## Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=2 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
```


## Qwen/Qwen3-VL-30B-A3B-Thinking-FP8
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=2 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-30B-A3B-Thinking-FP8

## Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=4 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

## Qwen/Qwen3-VL-235B-A22B-Thinking-FP8
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=4 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model Qwen3VL --model-name Qwen/Qwen3-VL-235B-A22B-Thinking-FP8


# Gemma3
## google/gemma-3-4b-it
```bash
srun --partition=${smallpartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model GEMMA3 --model-name google/gemma-3-4b-it
```

## google/gemma-3-12b-it
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model GEMMA3 --model-name google/gemma-3-12b-it
```

## google/gemma-3-27b-it
```bash
srun --partition=${largepartition} \
     --job-name=vllm-test \
     --nodes=1 \
     --export=ALL,HF_HUB_CACHE=/netscratch/thomas/models/,HF_XET_CACHE=/netscratch/thomas/huggingface-xet \
     --ntasks=1 \
     --cpus-per-task=6 \
     --gpus-per-task=1 \
     --mem-per-cpu=8G \
     --time=1-00:00:00 \
    python solveTask.py --model GEMMA3 --model-name google/gemma-3-27b-it
```