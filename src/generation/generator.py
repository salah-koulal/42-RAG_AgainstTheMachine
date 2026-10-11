"""Loads Qwen/Qwen3-0.6B, builds a prompt from retrieved sources, produces an answer (Task 5)."""
from src.models import MinimalSource
from src.io_utils import RAW_DIR
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch

def extract_source_text(source: MinimalSource) -> str:
    """Read file and slice character span for a MinimalSource."""
    try:
        full_path = RAW_DIR / source.file_path
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        # print(content[source.first_character_index:source.last_character_index])
        return content[source.first_character_index:source.last_character_index]
    except Exception:
        return ""



def build_context(sources: List[MinimalSource]) -> str:
    """Combine extracted tex from all sources into a unified context string."""

    context_blocks = []
    for idx, src in enumerate(sources, start=1):
        text = extract_source_text(src)
        if text.strip():
            context_blocks.append(f"[Document {idx} - {src.file_path}\n{text}]")
    return "\n\n".join(context_blocks)


class QwenGenerator:
    """Generates natural language answers grounded in retrieved codebase context."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)


    def generate_answer(self, query: str, sources: List[MinimalSource]) -> str:
        """Build context from sources, construct prompt, and generate answer."""
        context = build_context(sources)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. Answer the user question "
                    "accurately using ONLY the provided codebase context snippets. "
                    "If the answer cannot be deduced from the context, state that clearly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        # 1. Format messages into the model's chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )

        # 2. Tokenize the input string and push to device (CPU/GPU)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 3. Generate tokens
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # 4. Extract ONLY newly generated tokens (slice off the input prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]

        # 5. Decode back to human text
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer.strip()
