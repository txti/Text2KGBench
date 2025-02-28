import subprocess
import time
from typing import Any, Dict, Tuple, TypeVar

import litellm

from kgbench.utils.eval import calculate_metrics

Triple = TypeVar("Triple", bound=Tuple[str, str, str])

def get_llm_response(
        prompt: str,
        model: str,
        temperature: float = 0.0,
        prefix: str = "ollama",
        base_url: str = "http://localhost:11434",
        api_key="sk-1234") -> Dict[str, Any]:
    try:
        # Start timing
        start_time = time.time()

        # Generate response
        response = litellm.completion(
            model=f"{prefix}/{model}",
            temperature=temperature,
            api_key=api_key,
            api_base=base_url,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
        )

        # End timing
        end_time = time.time()

        # Extract response text
        response_text = response.choices[0].message.content

        # Calculate metrics
        metrics = calculate_metrics(
            start_time, end_time, response_text, prompt)

        return {"success": True, "response": response_text, "metrics": metrics}

    except Exception as e:
        return {"success": False, "error": str(e), "metrics": None}


def download_ollama_model(model_tag):
    subprocess.run(["ollama", "pull", model_tag])
