from gsme_q_mt.model_utils import CLAUDE_MODELS
from gsme_q_mt.model_utils import GPT_COSTS
from gsme_q_mt.model_utils import load_cache_file


class Evaluator:

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mc",
      model_role_name: str = "assistant",
      parallel_model_calls: bool = True,
      vllm_port: int = 8011,
      model_url: str = "",
      **kwargs,
  ):
    self.model_name = model_name
    self.generation_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384,
    }

    # model_url kept for backwards compatibility with gsm.py calling cached_generate(model_url=...)
    self.model_url = model_url or self.model_name

    if "gemini" in self.model_name:
      # Gemini SDK uses model name as identifier
      self.model_url = self.model_name
    elif self.model_name in GPT_COSTS:
      # Azure OpenAI chat.completions
      self.generation_config = {
          "max_completion_tokens": 16384,
      }
    elif self.model_name in CLAUDE_MODELS:
      self.generation_config = {
          "temperature": 0.0,
          "max_tokens": 16384,
      }
    elif "qwen" in self.model_name:
      if self.model_name == "qwen_30b":
        self.model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
      elif self.model_name == "qwen_4b":
        self.model_name = "Qwen/Qwen3-4B-Thinking-2507-FP8"
      elif self.model_name == "qwen_80b":
        self.model_name = "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"
      elif self.model_name == "qwen3_30b_instruct":
        self.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
      elif self.model_name == "qwen3-30b-instruct-2507":
        self.model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    elif self.model_name == "gpt_oss_20b":
      self.model_name = "openai/gpt-oss-20B"

    self.cache = cache
    self.cache_file = cache_file
    if cache is None and cache_file is not None:
      self.cache = load_cache_file(cache_file)
      print(f"Loaded {len(self.cache)} entries from {cache_file}")

    self.use_cot = use_cot
    self.fs_samples = fs_samples
    self.eval_mode = eval_mode
    self.model_role_name = model_role_name

    # Needed by gsm.py when calling cached_generate(..., parallel_model_calls=self.parallel_model_calls)
    self.parallel_model_calls = parallel_model_calls

    self.vllm_port = vllm_port
    self.use_invalid_facts_sets = kwargs.get("use_invalid_facts_sets", False)
