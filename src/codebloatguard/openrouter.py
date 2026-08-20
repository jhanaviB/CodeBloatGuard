"""
OpenRouter/Omniroute provider for judge, triage, and conventions.

Routes through OpenRouter API which can access multiple providers including
Gemini, Claude, and others. This allows using different API keys and quotas.
"""

import json
import os

import requests


def generate_json(model: str, prompt: str, schema: dict) -> dict:
    """Call OpenRouter API with structured output request.
    
    OpenRouter doesn't natively support response_schema like Gemini, so we
    ask for JSON in the prompt and validate it ourselves.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Build the instruction for JSON output
    required_keys = schema.get("required", [])
    required_str = ", ".join(f'"{k}"' for k in required_keys)
    
    json_instruction = f"""
Reply with ONLY a valid JSON object. No markdown, no code fences, no explanations.

Required keys: {required_str}

Schema:
{json.dumps(schema, indent=2)}

Your response must be parseable JSON matching this schema exactly.
"""
    
    # Combine prompt with JSON instruction
    full_prompt = f"{prompt}\n\n{json_instruction}"
    
    # Map Gemini model names to OpenRouter equivalents
    model_map = {
        "gemini-2.5-flash": "google/gemini-2.0-flash-exp:free",  # Free tier
        "gemini-2.0-flash": "google/gemini-2.0-flash-exp:free",
        # Add more mappings as needed
    }
    
    openrouter_model = model_map.get(model, model)
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": openrouter_model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}  # Request JSON mode
        },
        timeout=30
    )
    
    response.raise_for_status()
    result = response.json()
    
    # Extract the JSON from the response
    content = result["choices"][0]["message"]["content"]
    
    # Parse and validate
    try:
        data = json.loads(content.strip())
    except json.JSONDecodeError:
        # Try to extract JSON if wrapped in markdown
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            raise ValueError(f"Could not parse JSON from response: {content[:200]}")
    
    # Validate required keys
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in response: {data}")
    
    # Validate enum values
    for key, spec in schema.get("properties", {}).items():
        if "enum" in spec and key in data:
            if data[key] not in spec["enum"]:
                raise ValueError(
                    f"{key}={data[key]!r} not in allowed values {spec['enum']}"
                )
    
    return data
