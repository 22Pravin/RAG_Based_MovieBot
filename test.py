import openai
openai.api_key = "OPENAI_API_KEY"
try:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Hello",
        max_tokens=5
    )
    print("API key works!")
except Exception as e:
    print(f"API key error: {e}")
