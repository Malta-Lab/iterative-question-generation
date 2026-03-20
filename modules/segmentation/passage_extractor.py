from typing import List


class PassageExtractor:

    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)

        return chunks

    def run(self, context):
        passages = []

        for doc in context.documents:
            chunks = self.chunk_text(doc)

            for chunk in chunks:
                passages.append({
                    "text": chunk,
                    "source": doc  # opcional, mas útil pra debug
                })

        context.passages = passages
        return context