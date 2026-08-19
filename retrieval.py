"""Document chunking and search.

TF-IDF over word and bigram features. An earlier version used ChromaDB, but its
default embedding model downloads about 80MB on first use and that download hung
on a cold start, leaving the page blank. For a few dozen SOPs and policy files
TF-IDF is accurate enough, starts instantly and needs no network.

No Streamlit state here so it can be tested directly.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

# sklearn's English list misses inflections that are common in how staff phrase
# questions. Left in, "does" from "how long does the tour take" matched a
# document reading "does not expire" and outranked the one about tours.
QUESTION_NOISE = {
    "does", "did", "doing", "done", "doesn", "didn", "don",
    "isn", "aren", "wasn", "weren", "won", "wouldn", "couldn", "shouldn",
    "ll", "ve", "re", "im",
    "tell", "know", "ask", "asked", "asking", "say", "said",
    "please", "thanks", "thank", "hi", "hello", "hey",
    "need", "want", "like", "just", "got", "get", "gets",
    "allowed", "supposed", "okay", "ok", "yes", "no",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | QUESTION_NOISE)


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def chunk_document(text: str, source: str) -> list[Chunk]:
    """Split one document into overlapping chunks."""
    if not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return [
        Chunk(source=source, index=i, text=piece)
        for i, piece in enumerate(splitter.split_text(text))
    ]


class Index:
    """A searchable set of chunks.

    Rebuilds the matrix on every add. Wasteful at scale, irrelevant here, and it
    keeps the vocabulary consistent between indexing and querying.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(c.source for c in self.chunks))

    def add(self, chunks: list[Chunk]) -> int:
        """Add chunks, replacing any existing chunks from the same source."""
        if not chunks:
            return 0
        incoming = {c.source for c in chunks}
        self.chunks = [c for c in self.chunks if c.source not in incoming]
        self.chunks.extend(chunks)
        self._rebuild()
        return len(chunks)

    def remove_source(self, source: str) -> None:
        self.chunks = [c for c in self.chunks if c.source != source]
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.chunks:
            self._vectorizer, self._matrix = None, None
            return
        self._vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, question: str, k: int = 4, min_score: float = 0.02) -> list[Hit]:
        """Return the best matching chunks, strongest first.

        Hits below min_score are dropped so an unrelated question returns nothing
        instead of the least bad chunk in the corpus.
        """
        if self._vectorizer is None or not question.strip():
            return []

        scores = cosine_similarity(
            self._vectorizer.transform([question]), self._matrix
        )[0]
        ranked = sorted(
            (Hit(chunk=self.chunks[i], score=float(s)) for i, s in enumerate(scores)),
            key=lambda h: h.score,
            reverse=True,
        )
        return [h for h in ranked[:k] if h.score >= min_score]


def format_context(hits: list[Hit]) -> str:
    """Render hits as cited context for the model."""
    return "\n---\n".join(
        f"[Source: {h.chunk.source}, chunk {h.chunk.index}]\n{h.chunk.text}"
        for h in hits
    )
