from abc import ABC, abstractmethod

from huggingface_hub.errors import HFValidationError
from transformers import AutoTokenizer
from vllm import LLM
import torch
import os

class BaseLLM(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.trust_remote_code = kwargs.get("trust_remote_code", True)
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.tokenizer_mode = kwargs.get("tokenizer_mode", "slow")
        self.dtype = kwargs.get("dtype", "bfloat16")
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = kwargs.get("stop_token_ids", None)

        self.num_gpus = torch.cuda.device_count()
        self.llm = None # Language model instance

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def construct_prompt(self, question: str) -> str:
        pass

    def query(self, question: str) -> tuple:
        """
            Query the loaded language model with a specific question.

            This method constructs a usable prompt from the provided question and returns the
            necessary components for model inference. It ensures that the model is
            properly loaded before proceeding.

            Parameters:
                question (str): The input query or question to be processed by the model.

            Returns:
                tuple: A tuple containing:
                    - llm: The loaded language model instance.
                    - prompt (str): The constructed prompt specific to the model.
                    - stop_token_ids (list or None): A list of stop token IDs to define
                      end-of-sequence conditions, or None if not specified.
            """
        if self.llm is None:
            raise ValueError(f"Model {self.model_name} is not loaded. Call load_model() first.")
        prompt = self.construct_prompt(question)
        return self.llm, prompt, self.stop_token_ids

    def get_model_name(self) -> str:
        return self.model_name

    def get_stop_token_ids(self) -> list:
        return self.stop_token_ids

class AriaLLM(BaseLLM):

    def __init__(self, model_name: str = "rhymes-ai/Aria", **kwargs):
        super().__init__(model_name, **kwargs)

        #Override the default stop token ids
        self.stop_token_ids = [93532, 93653, 944, 93421, 1019, 93653, 93519]
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            tokenizer_mode=self.tokenizer_mode,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            trust_remote_code=self.trust_remote_code,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
        )

    def construct_prompt(self, question: str) -> str:
        return (f"<|im_start|>user\n<fim_prefix><|img|><fim_suffix>\n{question}"
                "<|im_end|>\n<|im_start|>assistant\n")

class ChameleonLLM(BaseLLM):
    def __init__(self, model_name: str = "facebook/chameleon-7b", **kwargs):
        super().__init__(model_name)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        # Initialize the LLM with the specified parameters.
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache
        )

    def construct_prompt(self, question: str) -> str:
        return f"{question}<image>"

#Currently, does not work!
class DeepseekVL2(BaseLLM):
    def __init__(self, model_name: str = "deepseek-ai/deepseek-vl2-tiny", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = None  # Set stop_token_ids to None for Deepseek-VL2 models.
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.hf_overrides = {"architectures": ["DeepseekVLV2ForCausalLM"]}
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            hf_overrides=self.hf_overrides
        )

    def construct_prompt(self, question: str) -> str:
        return f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"

# Gemma 3
class Gemma3(BaseLLM):
    def __init__(self, model_name: str = "google/gemma-3-4b-it", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 2048)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
#        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.stop_token_ids = None
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            mm_processor_kwargs={"do_pan_and_scan": True},# Default is False; setting it to True is not supported in V1 yet
#            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
    )

    def construct_prompt(self, question: str) -> str:
        return ("<bos><start_of_turn>user\n"
                f"<start_of_image>{question}<end_of_turn>\n"
                "<start_of_turn>model\n")

class GLM4V(BaseLLM):
    def __init__(self, model_name: str = "THUDM/glm-4v-9b", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 2048)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = kwargs.get("stop_token_ids", [151329, 151336, 151338])
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.hf_overrides = '{"architectures": ["GLM4VForCausalLM"]}'
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            trust_remote_code=True,
            enforce_eager=True,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            hf_overrides=self.hf_overrides
        )

    def construct_prompt(self, question: str) -> str:
        return question

# Idefics3-8B-Llama3
class Idefics3(BaseLLM):
    def __init__(self, model_name: str = "HuggingFaceM4/Idefics3-8B-Llama3", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.enforce_eager = kwargs.get("enforce_eager", True)
        self.mm_processor_kwargs = kwargs.get("mm_processor_kwargs", {
            "size": {
                "longest_edge": 3 * 364
            },
        })
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=self.enforce_eager,
        # if you are running out of memory, you can reduce the "longest_edge".
        # see: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3#model-optimizations
            mm_processor_kwargs=self.mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
        )

    def construct_prompt(self, question: str) -> str:
        prompt = (
            f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
        )
        return prompt

# InternVL
class InternVL(BaseLLM):
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-2B", max_model_len: int = 4096, disable_mm_preprocessor_cache: bool = False):
        super().__init__(model_name)
        self.max_model_len = max_model_len
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        self.stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None] #Remove None element
        self.load_model()

    def load_model(self):
        # Initialize the LLM using the model name and relevant parameters.
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
        )

    def construct_prompt(self, question: str) -> str:
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt

class LLAVA(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        super().__init__(model_name)
        self.load_model()

    def load_model(self):
        self.llm = LLM(model=self.model_name)

    def construct_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nASSISTANT:"

class LLAVANext(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", max_model_len: int = 4096):
        super().__init__(model_name)
        self.stop_token_ids = None  # Set stop_token_ids to None for LLAVA Next models.
        self.max_model_len = max_model_len
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=self.model_name, max_model_len=self.max_model_len, tensor_parallel_size=self.num_gpus)

    def construct_prompt(self, question: str) -> str:
        return f"[INST] <image>\n{question} [/INST]"

# LLaVA-OneVision
class OneVision(BaseLLM):
    def __init__(self, model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf", max_model_len : int =16384, disable_mm_preprocessor_cache : bool=False):
        super().__init__(model_name)
        self.stop_token_ids = None
        self.max_model_len = max_model_len
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(model=self.model_name,
                       max_model_len=self.max_model_len,
                       disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
                       tensor_parallel_size=self.num_gpus)

    def construct_prompt(self, question: str) -> str:
        return  f"<|im_start|>user <image>\n{question}<|im_end|> \
                <|im_start|>assistant\n"


# LLama 3.2
class LLAMA(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", max_model_len: int = 4096, max_num_seqs=16, disable_mm_preprocessor_cache=False):
        super().__init__(model_name)
        self.stop_token_ids = None  # Set stop_token_ids to None for LLAVA Next models.
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=1024,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            tensor_parallel_size=self.num_gpus,
            enforce_eager=True      )

    def construct_prompt(self, question: str) -> str:
        messages = [{
            "role":
                "user",
            "content": [{
                "type": "image"
            }, {
                "type": "text",
                "text": f"{question}"
            }]
        }]
        prompt = self.tokenizer.apply_chat_template(messages,
                                               add_generation_prompt=True,
                                               tokenize=False)
        return prompt

class LLAMA4(BaseLLM):
    def __init__(self, model_name: str = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",max_model_len=8192, max_num_seqs=4, **kwargs):
        super().__init__(model_name)
        self.stop_token_ids = None  # Set stop_token_ids to None for LLAVA Next models.
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (TypeError, OSError, HFValidationError): #Because the GGUF tokenizer seems not to work
            print(f"Failed to load tokenizer for {model_name}, using base Llama tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16")
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            tensor_parallel_size=self.num_gpus,
            enforce_eager=True      )

    def construct_prompt(self, question: str) -> str:
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": f"{question}"}],
                }
            ]
        ]
        prompts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts = prompts[0] #Convert from array to regular string
        return prompts

class H2OVL_Mississippi(BaseLLM):
    def __init__(self, model_name: str = "h2oai/h2ovl-mississippi-800m", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.stop_token_ids = [self.tokenizer.eos_token_id]
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            trust_remote_code=True
        )

    def construct_prompt(self, question: str) -> str:
        messages = [{"role": "user", "content": f"<image>\n{question}"}]

        prompts = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompts


class Molmo(BaseLLM):
    def __init__(self, model_name: str = "allenai/Molmo-7B-D-0924", **kwargs):
        super().__init__(model_name, **kwargs)
        self.stop_token_ids = None  # Set stop_token_ids to None as it's not specified
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=self.num_gpus,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache
        )

    def construct_prompt(self, question: str) -> str:
        return f"<|im_start|>user <image>\n{question}<|im_end|> <|im_start|>assistant\n"

class NVLM(BaseLLM):
    def __init__(self, model_name: str = "nvidia/NVLM-D-72B", max_model_len: int = 4096, disable_mm_preprocessor_cache: bool = False, tensor_parallel_size: int=4, **kwargs):
        super().__init__(model_name, **kwargs)
        self.stop_token_ids = None  # Set stop_token_ids to None as it's not specified
        self.max_model_len = max_model_len
        self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache
        self.tensor_parallel_size = tensor_parallel_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            enforce_eager=True
        )

    def construct_prompt(self, question: str) -> str:
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        return prompt


class PaliGemma2(BaseLLM):
    def __init__(self, model_name: str = "google/paligemma2-3b-ft-docci-448", **kwargs):
        super().__init__(model_name, **kwargs)
        self.stop_token_ids = None  # Set stop_token_ids to None as it's not specified
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()

    def load_model(self):
        self.llm = LLM(model=self.model_name, disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache)

    def construct_prompt(self, question: str) -> str:
        return "caption en"


class Phi3VisionLLM(BaseLLM):
    def __init__(self, model_name: str = "microsoft/Phi-3.5-vision-instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.mm_processor_kwargs = kwargs.get("mm_processor_kwargs", {"num_crops": 16})
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        # Initialize the LLM using the model name and relevant parameters.
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
        )

    def construct_prompt(self, question: str) -> str:
        return f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

class PixtralHF(BaseLLM):
    def __init__(self, model_name: str = "mistral-community/pixtral-12b", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
        )

    def construct_prompt(self, question: str) -> str:
        return f"<s>[INST]{question}\n[IMG][/INST]"


class Qwen2VL(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        super().__init__(model_name)
        self.stop_token_ids = None

        self.max_model_len = 4096
        self.max_num_seqs = 1
        self.tokenizer_mode = "auto"
        self.gpu_memory_utilization = 0.9
        # self.mm_processor_kwargs={"size": {"shortest_edge": 318, "longest_edge": 450}}
        self.mm_processor_kwargs = {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "size": {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280},
        }
        self.limit_mm_per_prompt = {"image": 1}

        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=self.trust_remote_code,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            tokenizer_mode=self.tokenizer_mode,
            dtype=self.dtype,
        )

    def construct_prompt(self, question):
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

class Qwen2_5_VL(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", max_num_seqs: int = 5, **kwargs):
        self.max_num_seqs = max_num_seqs
        self.stop_token_ids = None

        self.limit_mm_per_prompt = {"image": 1}
        self.mm_processor_kwargs = {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
            "truncation": False,
            "size": {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280},
        }
        super().__init__(model_name)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            mm_processor_kwargs=self.mm_processor_kwargs,
            limit_mm_per_prompt=self.limit_mm_per_prompt
        )

    def construct_prompt(self, question: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

class QWEN2_5_OMNI(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B", max_num_seqs: int = 5, **kwargs):
        super().__init__(model_name)
        self.max_num_seqs = max_num_seqs
        self.stop_token_ids = None
        self.mm_processor_kwargs = mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": [1],
        }
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            mm_processor_kwargs=self.mm_processor_kwargs,
            limit_mm_per_prompt={"image": 1}
        )

    def construct_prompt(self, question: str) -> str:
        default_system = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
            "Group, capable of perceiving auditory and visual inputs, as well as "
            "generating text and speech."
        )
        placeholder = "<|IMAGE|>"

        return(
            f"<|im_start|>system\n{default_system}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_bos|>{placeholder}<|vision_eos|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class Ovis2(BaseLLM):
    def __init__(self, model_name: str = "AIDC-AI/Ovis2-1B", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = None  # Set stop_token_ids to None for Ovis2 models.
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            trust_remote_code=True,
            dtype="half",
        )

    def construct_prompt(self, question: str) -> str:
        messages = [
            [{"role": "user", "content": f"<image>\n{question}"}]
        ]

        prompts = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts = prompts[0]  # Convert from array to regular string
        return prompts

class Qwen3VL(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-4B-Instruct", max_num_seqs: int = 5, **kwargs):
        self.max_num_seqs = max_num_seqs
        self.stop_token_ids = None

        self.limit_mm_per_prompt = {"image": 1}
        self.mm_processor_kwargs = {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        }
        super().__init__(model_name, **kwargs)
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            mm_processor_kwargs=self.mm_processor_kwargs,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
        )

    def construct_prompt(self, question: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


class Qwen3VL_MoE(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct", max_num_seqs: int = 5, **kwargs):
        self.max_num_seqs = max_num_seqs
        self.stop_token_ids = None

        self.limit_mm_per_prompt = {"image": 1}
        self.mm_processor_kwargs = {
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        }
        super().__init__(model_name, **kwargs)
        self.load_model()

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            tensor_parallel_size=self.num_gpus,
            trust_remote_code=True,
            mm_processor_kwargs=self.mm_processor_kwargs,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
        )

    def construct_prompt(self, question: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

class Kimi_VL(BaseLLM):
    def __init__(self, model_name: str = "moonshotai/Kimi-VL-A3B-Instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 4096)
        self.stop_token_ids = None  # Set stop_token_ids to None for Kimi-VL models.
        self.max_num_seqs = None
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            trust_remote_code=True,
        )

    def construct_prompt(self, question: str) -> str:
        return "<|im_user|>user<|im_middle|><|media_start|>image<|media_content|>"
        f"<|media_pad|><|media_end|>{question}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"

class Mistral3(BaseLLM):
    def __init__(self, model_name: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503", **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_model_len = kwargs.get("max_model_len", 8192)
        self.max_num_seqs = kwargs.get("max_num_seqs", 2)
        self.stop_token_ids = None  # Set stop_token_ids to None for Mistral-3 models.
        self.disable_mm_preprocessor_cache = kwargs.get("disable_mm_preprocessor_cache", False)
        self.load_model()  # Call load_model after all attributes are initialized.

    def load_model(self):
        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            trust_remote_code=True,
        )

    def construct_prompt(self, question: str) -> str:
        return f"<s>[INST]{question}\n[IMG][/INST]"

class MockLLM(BaseLLM):
    """Minimal mock implementation using BaseLLM interface"""

    def __init__(self, model_name="mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.stop_token_ids = kwargs.get("stop_token_ids", [0, 1])
        self.load_model()

    def load_model(self):
        class DummyGenerator:
            def __init__(self, model="mock-model", **kwargs):
                self.model = model

            def generate(self, prompts, sampling_params=None, **kwargs):
                default_response = "Default mock response"
                if isinstance(prompts, str):
                    prompts = [prompts]

                results = []
                for prompt in prompts:
                    # Inline MockRequestOutput with nested MockGenerationOutput
                    output = type(
                        "MockRequestOutput",
                        (),
                        {
                            "prompt": prompt,
                            "finished": True,
                            "outputs": [
                                # Inline MockGenerationOutput
                                type(
                                    "MockGenerationOutput",
                                    (),
                                    {
                                        "text": default_response,
                                        "token_ids": list(range(10)),
                                    },
                                )
                            ],
                        },
                    )
                    results.append(output)

                return results
        self.llm = DummyGenerator(model=self.model_name)

    def construct_prompt(self, question: str) -> str:
        return f"Question: {question}\nAnswer:"


# Model mapping
model_mapping = {
    "ARIA": AriaLLM,
    "CHAMELEON": ChameleonLLM,
    "DEEPSEEKVL2": DeepseekVL2,
    "GEMMA": PaliGemma2,
    "GEMMA3": Gemma3,
    "GLM4V": GLM4V,
    "H2OVL": H2OVL_Mississippi,
    "IDEFICS3": Idefics3,
    "INTERNVL": InternVL,
    "KIMI_VL": Kimi_VL,
    "LLAMA": LLAMA,
    "LLAMA4": LLAMA4,
    "LLAVA": LLAVA,
    "LLAVA-ONE-VISION": OneVision,
    "LLAVANEXT": LLAVANext,
    "MISTRAL3": Mistral3,
    "MOLMO": Molmo,
    # "NVLM": NVLM,
    "OVIS2": Ovis2,
    "PHI3": Phi3VisionLLM,
    "PIXTRAL": PixtralHF,
    "QWEN2VL": Qwen2VL,
    "QWEN2_5OMNI": QWEN2_5_OMNI,
    "QWEN2_5VL": Qwen2_5_VL,
    "QWEN3VL": Qwen3VL,
    "QWEN3VL_MOE": Qwen3VL_MoE,
}
