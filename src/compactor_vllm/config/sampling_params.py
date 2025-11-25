from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_new_tokens: int = 256

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("Temperature cannot be negative")
