# Retrieval for question answering

Retrieval-augmented generation (RAG) answers a question in two stages. First a retriever finds the
passages most relevant to the question. Then a language model writes an answer using only those
passages, which keeps the answer grounded and makes it possible to cite sources.

## BM25

BM25 is a ranking function from classical information retrieval. It scores a passage by how often
the query terms appear in it, weighted by how rare each term is across the whole collection
(inverse document frequency). Repeated terms add less and less score, controlled by the parameter
k1, and long passages are penalized relative to the average length, controlled by b. Typical values
are k1 = 1.5 and b = 0.75.

BM25 needs no training and no GPU, which makes it a strong baseline. It fails when the question and
the passage use different words for the same idea, for example "car" and "automobile".

## Dense retrieval

Dense retrievers embed questions and passages as vectors and rank by cosine similarity. They handle
synonyms and paraphrases better than BM25 but need an embedding model and a vector index. Many
systems combine both and merge the two rankings, which is called hybrid search.

## Chunking

Documents are split into chunks before indexing. Chunks that are too large dilute the relevant
sentence with unrelated text; chunks that are too small lose context. Splitting on headings and
paragraphs and keeping chunks to a few hundred words is a reasonable default.
