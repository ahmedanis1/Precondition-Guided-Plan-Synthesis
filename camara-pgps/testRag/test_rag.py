from pprint import pprint

from camara_pgps.retrieval.rag import get_rag_retriever
from camara_pgps.retrieval.retriever import get_retriever


QUERY = (
    "Show me the available QoS profiles that provide the highest "
    "quality and lowest latency for gaming"
)


def main():
    # ================================================================
    # 1. DIRECT VECTOR STORE / RETRIEVER DEBUG
    # ================================================================

    retriever = get_retriever()

    print("\n" + "=" * 70)
    print("VECTOR STORE DEBUG")
    print("=" * 70)

    print("Persist directory:", retriever.vector_store.persist_directory)
    print("Collection:", retriever.vector_store.collection_name)
    print("Vector count:", retriever.vector_store.count())

    print("\nQuery:")
    print(QUERY)

    result = retriever.retrieve(QUERY)

    print("\nRetrieved chunks:", len(result.chunks))

    for chunk in result.chunks:
        print("\n--- CHUNK ---")
        print("ID:", chunk.id)
        print("Score:", chunk.score)
        print("API:", chunk.api_name)
        print("Operation:", chunk.operation_id)

        print("\nMetadata:")
        pprint(chunk.metadata)

        print("\nText:")
        print(chunk.text[:500])

    # ================================================================
    # 2. RAG RETRIEVER TEST
    # ================================================================

    rag = get_rag_retriever()

    result = rag.retrieve(QUERY)

    print("\n" + "=" * 70)
    print("RAG OPERATIONS")
    print("=" * 70)

    if not result.operations:
        print("NO OPERATIONS FOUND")
    else:
        for operation in result.operations:
            print()
            print("Operation ID :", operation.operation_id)
            print("API          :", operation.api_name)
            print("Version      :", operation.api_version)
            print("Method       :", operation.method)
            print("Path         :", operation.path)
            print("Summary      :", operation.summary)
            print("Score        :", operation.score)

            print("\nRequest body:")
            pprint(operation.request_body)

            print("\nParameters:")
            pprint(operation.parameters)

            print("\nRelationships:")
            pprint(operation.relationships)

            print("\nSupporting chunks:")
            pprint(operation.supporting_chunks)

    # ================================================================
    # 3. OPERATION IDS
    # ================================================================

    print("\n" + "=" * 70)
    print("OPERATION IDS")
    print("=" * 70)

    pprint(result.operation_ids())

    # ================================================================
    # 4. RETRIEVED APIS
    # ================================================================

    print("\n" + "=" * 70)
    print("RETRIEVED APIS")
    print("=" * 70)

    pprint(result.apis)

    # ================================================================
    # 5. RAG DOCUMENTATION CHUNKS
    # ================================================================

    print("\n" + "=" * 70)
    print("RAG DOCUMENTATION")
    print("=" * 70)

    print(
        "Documentation chunks:",
        len(result.documentation.chunks),
    )

    for chunk in result.documentation.chunks:
        print()
        print("ID:", chunk.id)
        print("Score:", chunk.score)
        print("API:", chunk.api_name)
        print("Operation:", chunk.operation_id)

    # ================================================================
    # 6. RAG NOTES
    # ================================================================

    print("\n" + "=" * 70)
    print("RAG NOTES")
    print("=" * 70)

    pprint(result.notes)


if __name__ == "__main__":
    main()