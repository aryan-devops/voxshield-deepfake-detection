import statistics
from typing import List, Dict, Any
from app.config import settings

class AnalysisService:
    @staticmethod
    def aggregate_metrics(window_results: List[Dict]) -> Dict[str, Any]:
        """Calculates aggregate metrics from all window results."""
        if not window_results:
            return {}
            
        fake_probs = [w["fake_probability"] for w in window_results]
        total_windows = len(window_results)
        
        max_fake = max(fake_probs)
        avg_fake = sum(fake_probs) / total_windows
        median_fake = statistics.median(fake_probs)
        
        strong_windows = sum(1 for p in fake_probs if p >= settings.strong_fake_threshold)
        moderate_windows = sum(1 for p in fake_probs if p >= settings.moderate_fake_threshold and p < settings.strong_fake_threshold)
        
        strong_ratio = strong_windows / total_windows
        moderate_ratio = moderate_windows / total_windows
        
        return {
            "maximum_fake_probability": max_fake,
            "average_fake_probability": avg_fake,
            "median_fake_probability": median_fake,
            "total_windows": total_windows,
            "strong_fake_windows": strong_windows,
            "moderate_fake_windows": moderate_windows,
            "strong_window_ratio": strong_ratio,
            "moderate_window_ratio": moderate_ratio
        }
        
    @staticmethod
    def generate_assessment(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the final assessment based on aggregated metrics.
        The backend must explain the result using actual model evidence.
        """
        strong_ratio = metrics["strong_window_ratio"]
        max_fake = metrics["maximum_fake_probability"]
        avg_fake = metrics["average_fake_probability"]
        
        # Determine classification
        if strong_ratio > 0.15 or max_fake > 0.85:
            prediction = "LIKELY_AI_GENERATED"
            fake_prob = max(avg_fake * 1.5, max_fake * 0.9) # Weighted representation
        elif metrics["moderate_window_ratio"] > 0.2 or max_fake > settings.moderate_fake_threshold:
            prediction = "SUSPICIOUS"
            fake_prob = avg_fake * 1.2
        else:
            prediction = "LIKELY_REAL"
            fake_prob = avg_fake

        fake_prob = min(max(fake_prob, 0.0), 1.0)
        
        return {
            "prediction": prediction,
            "fake_probability": fake_prob,
            "real_probability": 1.0 - fake_prob,
            "confidence": fake_prob if prediction == "LIKELY_AI_GENERATED" else (1.0 - fake_prob if prediction == "LIKELY_REAL" else 0.5)
        }

    @staticmethod
    def generate_explanation(metrics: Dict[str, Any], assessment: Dict[str, Any], window_results: List[Dict]) -> Dict[str, Any]:
        """
        Generates evidence-based explanation without using LLMs.
        """
        prediction = assessment["prediction"]
        
        summary = ""
        if prediction == "LIKELY_AI_GENERATED":
            summary = (f"VoxShield classified this recording as likely AI-generated because "
                       f"multiple analyzed windows produced elevated synthetic-voice signals. "
                       f"The strongest window reached {metrics['maximum_fake_probability']*100:.1f}% fake probability.")
        elif prediction == "SUSPICIOUS":
            summary = (f"The recording contains mixed signals. Some windows showed elevated "
                       f"synthetic-voice probabilities while others were classified as real. "
                       f"The maximum fake probability was {metrics['maximum_fake_probability']*100:.1f}%.")
        else:
            summary = (f"VoxShield found predominantly real-voice signals across the analyzed windows. "
                       f"The maximum fake probability remained below the strong signal threshold.")
                       
        # Collect top evidence
        evidence = []
        # Sort windows by fake prob descending
        sorted_windows = sorted(window_results, key=lambda w: w["fake_probability"], reverse=True)
        
        for w in sorted_windows[:3]: # Top 3 most suspicious windows
            if w["fake_probability"] >= settings.moderate_fake_threshold:
                sig_type = "strong_window" if w["fake_probability"] >= settings.strong_fake_threshold else "moderate_window"
                desc = "Strong synthetic-voice signal detected in this window." if sig_type == "strong_window" else "Moderate synthetic-voice signal detected."
                evidence.append({
                    "type": sig_type,
                    "start_time": w["start_time"],
                    "end_time": w["end_time"],
                    "fake_probability": w["fake_probability"],
                    "description": desc
                })
                
        factors = [
            {
                "name": "Maximum fake probability",
                "value": metrics["maximum_fake_probability"],
                "interpretation": "Strong model signal in at least one analyzed window." if metrics["maximum_fake_probability"] > settings.strong_fake_threshold else "Peak signal remained low."
            },
            {
                "name": "Strong window ratio",
                "value": metrics["strong_window_ratio"],
                "interpretation": f"{metrics['strong_fake_windows']} of {metrics['total_windows']} windows crossed the strong-signal threshold."
            }
        ]
        
        return {
            "summary": summary,
            "evidence": evidence,
            "factors": factors
        }
