"""Lab 1: RAG with no cloud.

Every step of the RAG pipeline in one file, so you can see the chunks, the
vectors and the scores yourself. Run `python rag_lab.py -h` for the commands.
"""
import argparse
import re

import numpy as np
from sentence_transformers import SentenceTransformer

DOC = "policy.md"
MODEL_A = "all-MiniLM-L6-v2"   # 384 numbers per vector; used for the index
MODEL_B = "all-mpnet-base-v2"  # 768 numbers per vector; used only in 'mismatch'
TOP_K = 6


# ---------- Ingestion: load and chunk ----------
def load_chunks(path=DOC):
    """Split the document on '## ' headings: one chunk per section."""
    text = open(path, encoding="utf-8").read()
    chunks = []
    for block in re.split(r"\n(?=## )", text):
        if not block.startswith("## "):
            continue  # skips the '# title' line at the top
        lines = block.strip().splitlines()
        title = lines[0][3:].strip()
        audience, body = "all", []
        for line in lines[1:]:
            if line.lower().startswith("audience:"):
                audience = line.split(":", 1)[1].strip().lower()
            elif line.strip():
                body.append(line.strip())
        chunks.append({
            "id": f"policy-{len(chunks) + 1}",
            "title": title,
            "audience": audience,        # metadata used for security trimming
            "text": " ".join(body),
        })
    return chunks


# ---------- Ingestion: embed and index ----------
def build_index(chunks, model):
    """Embed each chunk. The 'index' here is just a matrix: one row per chunk."""
    texts = [f"{c['title']}. {c['text']}" for c in chunks]
    # normalize=True makes every vector length 1, so dot product = cosine similarity
    return model.encode(texts, normalize_embeddings=True)


# ---------- Query: retrieve ----------
def allowed(chunk, groups):
    if groups is None:          # no trimming at all (the unsafe default)
        return True
    return chunk["audience"] == "all" or chunk["audience"] in groups


def search(query, chunks, index, model, groups=None, k=TOP_K):
    qv = model.encode([query], normalize_embeddings=True)[0]
    scores = index @ qv                       # cosine similarity with every chunk
    ranked = sorted(zip(scores, chunks), key=lambda p: p[0], reverse=True)
    return [(float(s), c) for s, c in ranked if allowed(c, groups)][:k]


def keyword_search(query, chunks, k=TOP_K):
    """Plain word overlap, for comparison with vector search."""
    words = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3}
    scored = []
    for c in chunks:
        chunk_words = set(re.findall(r"[a-z]+", (c["title"] + " " + c["text"]).lower()))
        scored.append((len(words & chunk_words), c))
    return sorted(scored, key=lambda p: p[0], reverse=True)[:k]


# ---------- Query: augment ----------
def build_prompt(query, results):
    sources = "\n\n".join(f"[{c['id']}] {c['title']}\n{c['text']}" for _, c in results)
    return (
        "You are an order operations assistant.\n"
        "Answer ONLY from the sources below and cite the source id in brackets.\n"
        "If the sources do not contain the answer, say you don't know.\n\n"
        f"SOURCES:\n{sources}\n\n"
        f"QUESTION: {query}"
    )


# ---------- Commands ----------
def cmd_chunks(_args):
    for c in load_chunks():
        print(f"[{c['id']}] {c['title']}  (audience: {c['audience']}, {len(c['text'].split())} words)")
        print(f"    {c['text'][:90]}...")


def cmd_embed(_args):
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_A)
    index = build_index(chunks, model)
    print(f"Index shape: {index.shape}  ->  {index.shape[0]} chunks x {index.shape[1]} numbers each")
    print(f"First 8 numbers of '{chunks[0]['title']}':")
    print("   ", np.round(index[0][:8], 3))
    print("\nHow close is each section to 'Returns'? (1.0 = same meaning)")
    for c, v in zip(chunks, index):
        print(f"    {float(index[0] @ v):.3f}  {c['title']}")


def print_results(results):
    for score, c in results:
        print(f"    {score:.3f}  [{c['id']}] {c['title']}  (audience: {c['audience']})")


def parse_groups(raw):
    return None if raw is None else {g.strip().lower() for g in raw.split(",")}


def cmd_search(args):
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_A)
    index = build_index(chunks, model)
    groups = parse_groups(args.groups)
    print(f"Vector search for: {args.query!r}   user groups: {groups or 'NO TRIMMING'}")
    print_results(search(args.query, chunks, index, model, groups))


def cmd_keyword(args):
    print(f"Keyword search for: {args.query!r}  (score = shared words)")
    for score, c in keyword_search(args.query, load_chunks()):
        print(f"    {score}  [{c['id']}] {c['title']}")


def cmd_mismatch(_args):
    chunks = load_chunks()
    index = build_index(chunks, SentenceTransformer(MODEL_A))
    qv = SentenceTransformer(MODEL_B).encode(["Can I send it back?"], normalize_embeddings=True)[0]
    print(f"Index built with {MODEL_A}: {index.shape[1]} numbers per vector")
    print(f"Query embedded with {MODEL_B}: {qv.shape[0]} numbers")
    try:
        index @ qv
    except ValueError as e:
        print(f"\nSearch failed: {e}")
        print("Different models produce different coordinate systems. Even when the sizes")
        print("match, the numbers mean different things, so the scores would be garbage.")
        print("Rule: change the embedding model -> re-embed the whole index.")


def cmd_prompt(args):
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_A)
    index = build_index(chunks, model)
    results = search(args.query, chunks, index, model, parse_groups(args.groups))
    print("Retrieved:")
    print_results(results)
    print("\n--- Prompt that would be sent to the LLM ---\n")
    print(build_prompt(args.query, results))


def main():
    p = argparse.ArgumentParser(description="Lab 1: RAG with no cloud")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chunks", help="Step 1: show the chunks").set_defaults(fn=cmd_chunks)
    sub.add_parser("embed", help="Step 2: show the vectors").set_defaults(fn=cmd_embed)
    for name, fn, text in [("search", cmd_search, "Step 3: vector search"),
                           ("keyword", cmd_keyword, "Step 3: keyword search"),
                           ("prompt", cmd_prompt, "Step 6: build the grounded prompt")]:
        sp = sub.add_parser(name, help=text)
        sp.add_argument("query")
        if name != "keyword":
            sp.add_argument("--groups", help="comma-separated user groups, e.g. sales or hr")
        sp.set_defaults(fn=fn)
    sub.add_parser("mismatch", help="Step 4: query with the wrong model").set_defaults(fn=cmd_mismatch)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
