from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


print("\n========== CONTEXT BREAK DEMO ==========\n")

resp1 = llm.invoke(
    "We are building an AI system for processing medical insurance claims."
)

print("Response 1:")
print(resp1.content)

# This second question may fail or behave inconsistently because it is a separate LLM call.
# The model does not automatically remember the first prompt unless we pass conversation history.
# Without history, the phrase "this system" may be unclear, so the model may return a generic answer.
resp2 = llm.invoke(
    "What are the main risks in this system?"
)

print("\nResponse 2:")
print(resp2.content)


print("\n========== CONTEXT FIX USING MESSAGES API ==========\n")

messages = [
    SystemMessage(
        content="You are a senior AI architect reviewing production systems."
    ),
    HumanMessage(
        content="We are building an AI system for processing medical insurance claims."
    )
]

response1 = llm.invoke(messages)

print("Message API Response 1:")
print(response1.content)

messages.append(AIMessage(content=response1.content))

messages.append(
    HumanMessage(
        content="What are the main risks in this system?"
    )
)

response2 = llm.invoke(messages)

print("\nMessage API Response 2:")
print(response2.content)


print("\n========== REFLECTION ==========\n")

print(
    "Plain string-based LLM calls are stateless. Each invoke call only receives "
    "the current prompt, so the model may not understand references like "
    "'this system'. In production systems, this can cause inconsistent or "
    "incorrect behavior. Using the Messages API fixes this by passing structured "
    "conversation history with roles such as system, user, and assistant. This "
    "allows the model to answer based on prior context."
)