import os
import torch

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)

from app.config import settings


class ModelService:
    _model = None
    _processor = None
    _device = None
    _loaded = False

    @classmethod
    def _detect_device(cls):
        import sys
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        # Only attempt MPS on Darwin (Mac) to avoid Linux issues
        if sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
            
        return torch.device("cpu")

    @classmethod
    def initialize(cls):
        cls._device = cls._detect_device()
        model_dir = settings.voxshield_model_dir

        print()
        print("=" * 60)
        print("Loading VoxShield")
        print("=" * 60)
        print("Model directory:", model_dir)
        print("Device:", cls._device)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory does not exist: {model_dir}"
            )

        print("Loading feature extractor...")
        cls._processor = AutoFeatureExtractor.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        print("Loading model...")
        cls._model = AutoModelForAudioClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )

        cls._model.to(cls._device)
        cls._model.eval()

        cls._loaded = True

        print()
        print("✅ VoxShield model loaded successfully")
        print("Labels:", cls._model.config.id2label)
        print("=" * 60)

    @classmethod
    def get_status(cls):
        labels = cls._model.config.id2label if cls._model else {}
        return {
            "loaded": cls._loaded,
            "device": str(cls._device) if cls._device else "cpu",
            "model_name": "VoxShield",
            "processor": "AutoFeatureExtractor",
            "labels": labels
        }

    @classmethod
    def get_sampling_rate(cls):
        if cls._processor and hasattr(cls._processor, "sampling_rate"):
            return cls._processor.sampling_rate
        return 16000

    @classmethod
    def run_inference(cls, audio_samples):
        if not cls._loaded:
            raise RuntimeError("VoxShield model has not been loaded")

        sr = cls.get_sampling_rate()
        inputs = cls._processor(
            audio_samples,
            sampling_rate=sr,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(cls._device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = cls._model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())

        prediction = (
            "fake"
            if fake_probability >= real_probability
            else "real"
        )

        return real_probability, fake_probability, prediction