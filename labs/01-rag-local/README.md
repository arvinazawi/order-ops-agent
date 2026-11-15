# Lab 1: RAG with no cloud

Goal: see every step of RAG with your own eyes (the chunks, the vectors, the scores, the final prompt) before building it on Azure.
Time: about 1 hour. Cost: $0. Everything runs on your laptop.

## Where this lives
Put this folder in your order-ops agent repo as `labs/01-rag-local/`.
- It uses the same kind of policy document the agent will answer from.
- Lab 2 is the agent's real retrieval phase (Blob Storage, Azure AI Search, Azure OpenAI), so this lab is the warm-up for that phase.
- It shows on GitHub that you built RAG yourself, not just used it.

## Setup
```bash
cd labs/01-rag-local
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install sentence-transformers numpy
```
Add `.venv/` to the repo's `.gitignore`. The first run downloads a small embedding model (about 90 MB). After that, everything runs offline.

## Step 1: Chunk
```bash
python rag_lab.py chunks
```
Look at: 6 chunks, one per `##` heading, each with an id and an `audience` metadata field.
Write down: why split by heading instead of every 500 characters? (Hint: what happens to a rule that gets cut in half?)

## Step 2: Embed
```bash
python rag_lab.py embed
```
Look at: the index is a matrix of 6 rows × 384 numbers. One row is one chunk's "meaning coordinates".
Look at: how close each section is to Returns. Refunds should score higher than Goodwill credits.
Write down: the two highest and lowest scores, and why.

## Step 3: Vector search vs keyword search
```bash
python rag_lab.py keyword "Can I send it back?"
python rag_lab.py search  "Can I send it back?"
```
Look at: keyword search ranks "Damaged or wrong items" first, because it shares the words "send" and "back". Vector search should rank Returns first, because it matches on meaning.
Write down: this is why Azure AI Search uses **hybrid** search (keyword + vector, then a re-ranker). Each method catches what the other misses: keyword handles exact terms like order numbers and SKUs, and vector handles meaning.

## Step 4: Wrong embedding model (your weak spot from the quiz)
```bash
python rag_lab.py mismatch
```
Look at: the index was built with a 384-number model and the query used a 768-number model, so the search fails outright.
Write down: even when two models produce the same size, their coordinates mean different things. **Change the model → re-embed everything.**

## Step 5: Security trimming
```bash
python rag_lab.py search "Can I give a customer a 50 dollar credit?"
python rag_lab.py search "Can I give a customer a 50 dollar credit?" --groups sales
python rag_lab.py search "Can I give a customer a 50 dollar credit?" --groups hr
```
Look at: with no trimming, the internal HR section comes back to anyone. With `--groups sales` it's gone. With `--groups hr` it's back.
Write down: in Azure, `audience` becomes a filterable field in the index, and `--groups` comes from the user's Entra token.

## Step 6: Augment (build the grounded prompt)
```bash
python rag_lab.py prompt "My order came 6 days late. Can I get my shipping fee back?" --groups sales
```
Look at: the exact text the LLM would receive, which is the instructions, the sources with ids, and then the question. That's all "augmented generation" means.
In Lab 2, you'll send this prompt to your Azure OpenAI deployment and get back an answer that cites [policy-2].

## Experiments (pick two)
1. Ask something the policy doesn't cover: `search "Who is your CEO?"`. Look at how low the top score is. Why do real systems set a minimum score threshold?
2. Split the Refunds section into two `##` sections, re-run step 6, and see whether retrieval gets better or worse.
3. Change `TOP_K` to 1 and then 6 in `rag_lab.py`. What's the trade-off? (Too few sources: you miss the answer. Too many: noise and token cost.)

## You're done when you can explain out loud, in 2 minutes
- [ ] The two pipelines, and what happens in each step
- [ ] Why chunk boundaries matter
- [ ] What a vector is and what the index stores (vector + text + metadata)
- [ ] Why hybrid search beats vector-only or keyword-only
- [ ] Why the query must use the same embedding model as the index
- [ ] How security trimming works and where the user's groups come from

Then commit with a message like `lab 1: local RAG pipeline (chunk, embed, search, trim, augment)` and push.
