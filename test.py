from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# Quick check that the Groq key in .env works. The key itself is not printed.
print("GROQ_API_KEY found:", bool(os.getenv("GROQ_API_KEY")))

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

print(llm.invoke("Hello"))
