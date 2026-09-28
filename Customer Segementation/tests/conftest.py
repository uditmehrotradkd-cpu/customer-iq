import os

# Tests must never touch the real cloud account database or a real AI provider.
for name in ("SEG_DATABASE_URL", "DATABASE_URL", "SEG_LLM_API_KEY", "SEG_LLM_BASE_URL", "SEG_LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
    os.environ.pop(name, None)
