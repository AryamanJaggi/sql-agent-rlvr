"""Policy implementations: the callables passed into run_episode().

Two implementations:

  PromptedPolicy: local Ollama server (Qwen3-4B, GGUF quantization).
    Cheap, CPU-only, good for iterating on the harness itself, but not
    the same quantization SFT/GRPO train from.
  UnslothPolicy: the real unsloth/Qwen3-4B-unsloth-bnb-4bit weights
    via Unsloth's vLLM fast_generate backend, on GPU (Colab). What
    collect_sft_data.py and the real headroom calibration use. Also the
    actual weights the trained models are built from. 
"""

from __future__ import annotations

import ollama

PROMPTED_BASELINE_SYSTEM_PROMPT = """\
You are a careful SQL analyst agent. You answer natural-language \
questions about a SQLite database by exploring it and querying it - you \
never guess an answer without having run the query that proves it.

You have exactly three tools:
  inspect_schema  - lists every table and its columns. No input needed.
  run_sql         - executes a single read-only SELECT statement against \
the database and returns its result rows (or an error message if the \
query was invalid).
  final_answer    - ends the episode with your answer.

On every turn, respond with exactly ONE action, in exactly this format \
(no other text before or after):

Action: <tool_name>
Action Input: <input>

Work step by step: inspect the schema if you're unsure of table or \
column names, run a SELECT to check your reasoning, and only call \
final_answer once a run_sql result actually answers the question. The \
result of your last successful run_sql call is what gets checked \
against the correct answer - so don't call final_answer until your most \
recent query is the one you actually want to stand behind.

Example turn (a tool with no input - just omit the Action Input line):
Action: inspect_schema

Example turn:
Action: run_sql
Action Input: SELECT COUNT(*) FROM singer

Example final turn:
Action: final_answer
Action Input: 6
"""

# Used only by env/grpo_env.py's native-tool-calling path. 
# Same task framing as the ReAct prompt above, minus the 
# Action:/Action Input: formatting instructions. Native
# tool-calling presents the tool schema separately, so the model doesn't
# need text-format instructions for it.
GRPO_SYSTEM_PROMPT = """\
You are a careful SQL analyst agent. You answer natural-language \
questions about a SQLite database by exploring it and querying it - you \
never guess an answer without having run the query that proves it.

Use the available tools to inspect the schema and run SELECT queries. \
Work step by step: check table and column names if you're unsure, run a \
query to verify your reasoning, and only call final_answer once a query \
result actually answers the question. The result of your last successful \
query is what gets checked against the correct answer - so don't call \
final_answer until your most recent query is the one you actually want \
to stand behind.
"""


def _build_messages(transcript: str) -> list[dict]:
    """Pure formatting step, split out from the network call so it's
    testable without a running Ollama server.
    """
    return [
        {"role": "system", "content": PROMPTED_BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]


class PromptedPolicy:
    """Calls a local Ollama server. Requires `ollama serve` running with
    the target model already pulled.

    Ollama serves a GGUF quantization of Qwen3-4B, not the bnb-4bit/NF4
    quantization SFT/GRPO actually train from. Fine for iterating on
    the environment/harness locally, but for the real final comparison
    may introduce some error so use UnslothPolicy instead.
    """

    def __init__(self, model: str = "qwen3:4b", host: str | None = None):
        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model

    def __call__(self, transcript: str) -> str:
        response = self._client.chat(model=self.model, messages=_build_messages(transcript))
        return response["message"]["content"]


class UnslothPolicy:
    """The real base model: unsloth/Qwen3-4B-unsloth-bnb-4bit via
    Unsloth's vLLM fast_generate backend. Needs a GPU (Colab). unsloth
    and vllm are imported lazily here so import env.policies still
    works on a machine with neither installed (this one).

    enable_thinking=False: Qwen3's chat template defaults thinking mode
    ON, wraps every turn in a <think> block. Disabled to keep turns
    terse, matches how ReAct format was validated during calibration.

    lora_path: path to a trained adapter (train_sft.py / train_grpo.py
    output). None = untrained baseline, same weights no adapter - this
    flag lets a single class cover the baseline, SFT eval, and GRPO eval.
    """

    MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"

    def __init__(
        self,
        max_seq_length: int = 8192,
        gpu_memory_utilization: float = 0.7,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        lora_path: str | None = None,
    ):
        from unsloth import FastLanguageModel
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.MODEL_NAME,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        FastLanguageModel.for_inference(self.model)
        self._sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        # model.load_lora() isn't reliably available here - Unsloth only
        # attaches it as a bound method via get_peft_model()'s patching,
        # which a pure inference load (no fresh adapter being trained)
        # never calls, hence a real AttributeError hit during eval.
        # unsloth_zoo's own load_lora(load_tensors=False) branch does
        # nothing but this: vLLM's own LoRARequest(name, id, path) reads
        # the saved adapter directly off disk, no model-side LoRA
        # structure required.
        self._lora_request = LoRARequest("trained_adapter", 1, lora_path) if lora_path else None

    def __call__(self, transcript: str) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            _build_messages(transcript),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = self.model.fast_generate(
            prompt_text,
            sampling_params=self._sampling_params,
            lora_request=self._lora_request,
        )
        return output[0].outputs[0].text