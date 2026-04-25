from dotenv import load_dotenv
load_dotenv()

import os
print("KEY:", os.getenv("OPENROUTER_API_KEY"))  # debug

from llm import enhance_query

result = enhance_query("machine learning basics")
print("\nResult:\n", result)