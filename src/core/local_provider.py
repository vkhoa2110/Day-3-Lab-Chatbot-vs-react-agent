import time
import os
from typing import Dict, Any, Optional, Generator
from llama_cpp import Llama
from src.core.llm_provider import LLMProvider

class LocalProvider(LLMProvider):
    """
    LLM Provider for local models using llama-cpp-python.
    Optimized for CPU usage with GGUF models.
    """
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        max_tokens: int = 512,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        main_gpu: int = 0,
    ):
        """
        Initialize the local Llama model.
        Args:
            model_path: Path to the .gguf model file.
            n_ctx: Context window size.
            n_threads: Number of CPU threads to use. Defaults to all available.
            max_tokens: Maximum number of tokens to generate.
            n_gpu_layers: Number of layers to offload to GPU. -1 means offload all possible layers.
            n_batch: Prompt processing batch size.
            main_gpu: CUDA GPU index.
        """
        super().__init__(model_name=os.path.basename(model_path))
        self.max_tokens = max_tokens
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please download it first.")

        # n_threads=None will use all available cores
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            main_gpu=main_gpu,
            verbose=False
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # Phi-3 / Llama-3 style formatting if not handled by a template
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
        else:
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"

        response = self.llm(
            full_prompt,
            max_tokens=self.max_tokens,
            stop=["<|end|>", "Observation:"],
            echo=False
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        content = response["choices"][0]["text"].strip()
        usage = {
            "prompt_tokens": response["usage"]["prompt_tokens"],
            "completion_tokens": response["usage"]["completion_tokens"],
            "total_tokens": response["usage"]["total_tokens"]
        }

        return {
            "content": content,
            "usage": usage,
            "latency_ms": latency_ms,
            "provider": "local"
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"
        else:
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"

        stream = self.llm(
            full_prompt,
            max_tokens=self.max_tokens,
            stop=["<|end|>", "Observation:"],
            stream=True
        )

        for chunk in stream:
            token = chunk["choices"][0]["text"]
            if token:
                yield token
