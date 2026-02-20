"""
vLLM-based inference engine for batch generation.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from vllm import LLM, SamplingParams


@dataclass
class GenerationResult:
    """Result from a single generation."""
    prompt: str
    response: str
    finish_reason: str
    
    
class InferenceEngine:
    """Wrapper around vLLM for convenient inference."""
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 4,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.90,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_name: HuggingFace model name or path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            max_model_len: Maximum sequence length
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.model_short_name = model_name.split("/")[-1]
        
        print(f"Loading model: {model_name}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  Max model length: {max_model_len}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        print(f"Model loaded successfully: {self.model_short_name}")
    
    def _build_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a chat prompt from messages using the model's chat template.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt
        
        Returns:
            Formatted prompt string
        """
        full_messages = []
        
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        
        full_messages.extend(messages)
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            # Fallback for models without chat template
            print(f"Warning: Could not apply chat template: {e}")
            prompt = ""
            if system_prompt:
                prompt += f"System: {system_prompt}\n\n"
            for msg in messages:
                role = msg["role"].capitalize()
                prompt += f"{role}: {msg['content']}\n\n"
            prompt += "Assistant: "
        
        return prompt
    

    def _cap_max_tokens_for_batch(self, formatted_prompts, requested_max_tokens: int, reserve_tokens: int = 256) -> int:
        """Cap max_tokens so (prompt_tokens + max_tokens) <= max_model_len.

        Note: reserve_tokens can be overridden at runtime via env var
        `GALILEO_RESERVE_TOKENS` (int). This is useful for small-context models
        where long prompts (e.g., recovery prompts) can otherwise force
        `max_tokens` down to 1.

        vLLM SamplingParams uses a single max_tokens for the whole batch, so we
        choose the minimum safe value across the batch.
        """
        # Allow runtime override without changing runner code.
        try:
            reserve_tokens = int(os.environ.get("GALILEO_RESERVE_TOKENS", str(reserve_tokens)))
        except Exception:
            reserve_tokens = int(reserve_tokens)

        safe = int(requested_max_tokens)
        for p in formatted_prompts:
            try:
                prompt_tokens = len(self.tokenizer.encode(p))
            except Exception:
                prompt_tokens = max(1, len(p) // 4)
            avail = max(1, int(self.max_model_len) - int(prompt_tokens) - int(reserve_tokens))
            safe = min(safe, avail)
        if safe < requested_max_tokens:
            print(f"[warn] requested max_tokens={requested_max_tokens} capped to {safe} (max_model_len={self.max_model_len})")
        return max(1, safe)

    def generate_beam_search(
        self,
        prompts: List[str],
        n: int = 10,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> List[List[GenerationResult]]:
        """
        Generate responses using beam search-like sampling.
        
        Args:
            prompts: List of user prompts
            n: Number of completions per prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt to prepend
        
        Returns:
            List of lists of GenerationResults (n results per prompt)
        """
        # Build chat prompts
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompts.append(self._build_chat_prompt(messages, system_prompt))
        
        safe_max_tokens = self._cap_max_tokens_for_batch(formatted_prompts, max_tokens)

        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=safe_max_tokens,
            stop=["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"],
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        results = []
        for output in outputs:
            prompt_results = []
            for completion in output.outputs:
                prompt_results.append(GenerationResult(
                    prompt=output.prompt,
                    response=completion.text,
                    finish_reason=completion.finish_reason,
                ))
            results.append(prompt_results)
        
        return results
    
    def generate_greedy(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> List[GenerationResult]:
        """
        Generate responses using greedy decoding.
        
        Args:
            prompts: List of user prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt to prepend
        
        Returns:
            List of GenerationResults (one per prompt)
        """
        results = self.generate_beam_search(
            prompts=prompts,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        return [r[0] for r in results]
    
    def generate_multi_turn(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: float = 1.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> List[GenerationResult]:
        """
        Generate responses for multi-turn conversations.
        
        Args:
            conversations: List of conversations, each is a list of messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: System prompt to prepend
        
        Returns:
            List of GenerationResults
        """
        formatted_prompts = []
        for messages in conversations:
            formatted_prompts.append(self._build_chat_prompt(messages, system_prompt))
        
        safe_max_tokens = self._cap_max_tokens_for_batch(formatted_prompts, max_tokens)

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=safe_max_tokens,
            stop=["<|eot_id|>", "<|end|>", "</s>", "<|im_end|>"],
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        
        results = []
        for output in outputs:
            results.append(GenerationResult(
                prompt=output.prompt,
                response=output.outputs[0].text,
                finish_reason=output.outputs[0].finish_reason,
            ))
        
        return results
    
    def __del__(self):
        """Cleanup when engine is destroyed."""
        if hasattr(self, 'llm'):
            del self.llm
