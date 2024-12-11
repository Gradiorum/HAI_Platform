import json
import os

from .engines.pytorch_engine import PyTorchEngine
from .engines.vllm_engine import VLLMEngine
from .engines.deepseek_engine import DeepSeekEngine

from .test_time_compute.dense_verifier import DenseVerifier
from .test_time_compute.adaptive_distribution import AdaptiveDistribution
from .test_time_compute.entropix import EntropixSampler
from .test_time_compute.mcts import MCTSSampler
from .test_time_compute.deepseek import DeepSeekCompute

class InferenceManager:
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_config.json")) as f:
            self.model_config = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "backend_config.json")) as f:
            self.backend_config = json.load(f)

        self.verifier = DenseVerifier()
        self.adaptive = AdaptiveDistribution()
        self.entropix = EntropixSampler()
        self.mcts = MCTSSampler()
        self.deepseek_comp = DeepSeekCompute()

    def get_engine(self, model_name):
        model_info = self.model_config["models"].get(model_name, None)
        if model_info is None:
            raise ValueError(f"Model {model_name} not found in config")

        backend = model_info["backend"]
        if backend == "pytorch":
            return PyTorchEngine(
                model_name=model_info["model_name"],
                max_tokens=self.backend_config["max_tokens"],
                temperature=self.backend_config["temperature"],
                top_k=self.backend_config["top_k"],
                top_p=self.backend_config["top_p"]
            )
        elif backend == "vllm":
            return VLLMEngine(
                model_name=model_info["model_name"],
                max_tokens=self.backend_config["max_tokens"],
                temperature=self.backend_config["temperature"],
                top_k=self.backend_config["top_k"],
                top_p=self.backend_config["top_p"]
            )
        else:
            return DeepSeekEngine(model_info["model_name"])

    def run_inference(self, prompt, model_name, method="none"):
        engine = self.get_engine(model_name)
        if method == "none":
            return engine.generate(prompt)
        elif method == "dense_verifier":
            candidates = [engine.generate(prompt) for _ in range(3)]
            scores = self.verifier.score(prompt, candidates)
            best_idx = scores.index(max(scores))
            return candidates[best_idx]
        elif method == "adaptive_distribution":
            initial = engine.generate(prompt)
            return self.adaptive.refine(prompt, engine, initial)
        elif method == "entropix":
            return self.entropix.sample(prompt, engine)
        elif method == "mcts":
            return self.mcts.search(prompt, engine)
        elif method == "deepseek":
            return self.deepseek_comp.extended_thinking(prompt, engine)
        else:
            return engine.generate(prompt)
