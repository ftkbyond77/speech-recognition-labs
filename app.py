import re


class MockASRModel:
    """A tiny mock ASR model for demo and learning purposes."""

    def __init__(self):
        self.known_phrases = {
            "hello world": "greeting",
            "turn on the light": "turn_on_light",
            "turn off the light": "turn_off_light",
            "play music": "play_music",
            "stop music": "stop_music",
            "open the door": "open_door",
            "close the door": "close_door",
        }

    def normalize(self, text):
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

    def recognize(self, text):
        if text is None:
            return {"text": "", "intent": "empty", "confidence": 0.0}

        cleaned = self.normalize(str(text))
        if not cleaned:
            return {"text": "", "intent": "empty", "confidence": 0.0}

        best_match = None
        for phrase, intent in self.known_phrases.items():
            if cleaned == phrase:
                score = 1.0
            elif phrase in cleaned:
                score = 0.7
            else:
                score = 0.0

            if score > 0 and (best_match is None or score > best_match[0]):
                best_match = (score, intent, phrase)

        if best_match is None:
            return {"text": cleaned, "intent": "unknown", "confidence": 0.0}

        score, intent, _ = best_match
        return {
            "text": cleaned,
            "intent": intent,
            "confidence": round(score, 2),
        }

    def predict(self, audio_input):
        if isinstance(audio_input, (list, tuple)):
            audio_input = " ".join(str(item) for item in audio_input)
        return self.recognize(audio_input)


if __name__ == "__main__":
    model = MockASRModel()
    samples = [
        "hello world",
        "turn on the light",
        "play music",
        "close the door",
        "random command",
    ]

    for sample in samples:
        result = model.predict(sample)
        print(f"Input: {sample} -> {result}")
