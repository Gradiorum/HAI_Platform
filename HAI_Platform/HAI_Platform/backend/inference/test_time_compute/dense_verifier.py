class DenseVerifier:
    def __init__(self, verifier_model_name="some-scoring-model"):
        pass

    def score(self, prompt, candidates):
        scores = []
        for c in candidates:
            # Simple length-based scoring placeholder
            scores.append(len(c))
        return scores
