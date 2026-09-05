from camara_pgps.agents.intent_agent import IntentAgent
from camara_pgps.retrieval.rag import RagRetriever, _OPERATION_DOCTYPES

intent_agent = IntentAgent.from_settings()
intent = intent_agent.understand("increase my internet speed if i am at frankfurt")

print("goal:  ", intent.goal)
print("action:", intent.action)
print("entities:", intent.entities)
print()

rag = RagRetriever.from_settings()
queries = rag._build_queries(intent)
print("=== queries built from this intent ===")
for q in queries:
    print(" -", repr(q))
print()

semantic_chunks = rag._retrieve_chunks(queries, rag.top_k, None)
print(f"=== stage 1: semantic_chunks (post-fix, untruncated) count={len(semantic_chunks)} ===")

api_scores = {}
for chunk in semantic_chunks:
    if not chunk.api_name:
        continue
    doctype = str(chunk.metadata.get("document_type") or chunk.metadata.get("chunk_type") or "")
    weight = 1.0 if doctype in _OPERATION_DOCTYPES else 0.5
    api_scores[chunk.api_name] = api_scores.get(chunk.api_name, 0.0) + chunk.score * weight

ranked_apis = sorted(api_scores.items(), key=lambda kv: kv[1], reverse=True)
print("=== stage-1 api_scores (this is what gates which APIs get stage-2 structural expansion, top",
      rag.max_candidate_apis, "proceed) ===")
for api, score in ranked_apis:
    print(f"  {score:.4f}  {api}")
print()
candidate_apis = [api for api, _ in ranked_apis[: rag.max_candidate_apis]]
print("candidate_apis actually passed to stage 2:", candidate_apis)
print("Quality-On-Demand made the cut:", "Quality-On-Demand" in candidate_apis)
print()

ctx = rag.retrieve(intent)
print("=== final ctx.operations ===")
for op in sorted(ctx.operations, key=lambda o: -o.score):
    print(f"  {op.score:.4f}  {op.api_name:30s} {op.operation_id}")
print()
print("=== final ctx.apis (ranked by TOTAL evidence incl. non-operation chunks - NOT the stage-2 gate) ===")
print(ctx.apis)
