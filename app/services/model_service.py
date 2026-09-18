import os
import torch

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)

from app.config import settings


class ModelService:

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = self._detect_device()
        self.loaded = False

    def _detect_device(self):

        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    def load(self):

        model_dir = settings.VOXSHIELD_MODEL_DIR

        print()
        print("=" * 60)
        print("Loading VoxShield")
        print("=" * 60)
        print("Model directory:", model_dir)
        print("Device:", self.device)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory does not exist: {model_dir}"
            )

        print("Loading feature extractor...")

        self.processor = AutoFeatureExtractor.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        print("Loading model...")

        self.model = AutoModelForAudioClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.loaded = True

        print()
        print("✅ VoxShield model loaded successfully")
        print("Labels:", self.model.config.id2label)
        print("=" * 60)

    def is_loaded(self):
        return self.loaded

    def predict(self, audio, sampling_rate):

        if not self.loaded:
            raise RuntimeError(
                "VoxShield model has not been loaded"
            )

        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        real_probability = float(
            probabilities[0].item()
        )

        fake_probability = float(
            probabilities[1].item()
        )

        prediction = (
            "fake"
            if fake_probability >= real_probability
            else "real"
        )

        return {
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "prediction": prediction,
        }


# ============================================================
# GLOBAL VOXSHIELD MODEL SERVICE
# ============================================================

model_service = ModelService()