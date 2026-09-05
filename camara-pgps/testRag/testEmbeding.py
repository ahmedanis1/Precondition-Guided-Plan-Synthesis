from camara_pgps.retrieval.rag import RagRetriever

retriever = RagRetriever.from_settings()
ctx = retriever.retrieve("increase my internet speed if i am at frankfurt")
for op in sorted(ctx.operations, key=lambda o: -o.score):
    print(f"{op.score:.4f}  {op.api_name:30s} {op.operation_id}")