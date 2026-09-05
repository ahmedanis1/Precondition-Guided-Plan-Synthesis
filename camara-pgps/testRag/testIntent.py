from camara_pgps.agents.intent_agent import IntentAgent
from camara_pgps.retrieval.rag import RagRetriever

intent_agent = IntentAgent.from_settings()
intent = intent_agent.understand("increase my internet speed if i am at frankfurt")

print("goal:      ", intent.goal)
print("action:    ", intent.action)
print("text:      ", intent.text)
print("entities:  ", intent.entities)
print("constraints:", intent.constraints)
print()

retriever = RagRetriever.from_settings()
ctx = retriever.retrieve(intent)
for op in sorted(ctx.operations, key=lambda o: -o.score):
    print(f"{op.score:.4f}  {op.api_name:30s} {op.operation_id}")
