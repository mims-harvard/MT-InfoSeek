"""Base class for evaluators."""

from model_utils import CLAUDE_MODELS, GPT_COSTS, load_cache_file, MAX_TOKS
import model_registry

class Evaluator:
  """Base class for evaluators.

  Attributes:
    model_name: name of LLM to evaluate
    generation_config: generation config for LLM
    cache: cache of LLM responses
    cache_file: cache file of LLM responses
    use_cot: whether to use CoT or not
    fs_samples: number of few-shot samples to use
    eval_mode: evaluation mode, one of "mc", "isambig", "fullinfo"
    model_role_name: role name for the model
    vllm_port: port for the VLLM server
  """

  def __init__(
      self,
      model_name: str,
      cache=None,
      cache_file=None,
      use_cot: bool = False,
      fs_samples: int = 0,
      eval_mode: str = "mc",
      model_role_name: str = "assistant",
      vllm_port: int = 8011,
      **kwargs,
  ):
    self.model_name = model_name
    self.generation_config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": MAX_TOKS,
    }
    
    # Model-specific generation_configs
    if "gemini" in self.model_name:
      del self.generation_config["max_tokens"]
      self.generation_config["max_output_tokens"] = MAX_TOKS
    elif self.model_name in GPT_COSTS:
      # NOTE: GPT-5 and beyond don't allow for setting temperature or top_p
      del self.generation_config["max_tokens"]
      self.generation_config = {
          "max_completion_tokens": MAX_TOKS,
      }
      if self.model_name in {"gpt-5.2", "gpt-5.2-mini", "gpt-5.4", "gpt-5.4-mini"}:
        self.generation_config["reasoning_effort"] = kwargs.get("reasoning_effort", "medium")
    elif self.model_name in CLAUDE_MODELS:
      pass
    elif "qwen3" in self.model_name.lower():
      # including Qwen3 and Qwen3.5
      pass
    elif "gpt-oss" in self.model_name:
      self.generation_config["reasoning_effort"] = kwargs.get("reasoning_effort", "medium")
    elif "glm" in self.model_name.lower():
      pass
    elif "step" in self.model_name.lower():
      pass
    elif model_registry.ensure_local_registered(self.model_name):
      # A local vLLM chat model, registered or inferred from VLLM_PORT: use the
      # default sampling config (temperature/top_p/max_tokens), like Qwen.
      pass
    else:
      raise ValueError(
          f"Model {self.model_name} not supported. If it is a local model, start "
          f"its vLLM server and set VLLM_PORT (then it is inferred automatically).")

    max_tokens_override = kwargs.get("max_tokens")
    if max_tokens_override is not None:
      if "max_output_tokens" in self.generation_config:
        self.generation_config["max_output_tokens"] = max_tokens_override
      elif "max_completion_tokens" in self.generation_config:
        self.generation_config["max_completion_tokens"] = max_tokens_override
      else:
        self.generation_config["max_tokens"] = max_tokens_override
    
    self.cache = cache
    self.cache_file = cache_file
    if cache is None and cache_file is not None:
      self.cache = load_cache_file(cache_file)
      print(f"Loaded {len(self.cache)} entries from {cache_file}")
    self.use_cot = use_cot
    self.fs_samples = fs_samples
    self.eval_mode = eval_mode
    self.model_role_name = model_role_name
    self.vllm_port = vllm_port
    
    self.forbid_alternatives = kwargs.get("forbid_alternatives", False)
