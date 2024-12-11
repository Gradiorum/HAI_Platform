class EntropixSampler:
    def sample(self, prompt, engine):
        return engine.generate(prompt) + " [Entropix adjustments]"
