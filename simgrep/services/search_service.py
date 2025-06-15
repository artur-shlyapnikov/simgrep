from typing import Dict, List, Optional

from simgrep.core.abstractions import Embedder, Repository, VectorIndex
from simgrep.core.models import SearchResult


class SearchService:
    def __init__(
        self,
        store: Repository,
        embedder: Embedder,
        index: VectorIndex,
    ):
        self.store = store
        self.embedder = embedder
        self.index = index

    def search(self, query: str, *, k: int, min_score: float, file_filter: Optional[List[str]], keyword_filter: Optional[str]) -> List[SearchResult]:
        if len(self.index) == 0:
            return []

        query_embedding = self.embedder.encode([query], is_query=True)
        search_matches = list(self.index.search(query_embedding, k=k))

        if not search_matches:
            return []

        label_to_score: Dict[int, float] = {m.label: m.score for m in search_matches}
        usearch_labels = list(label_to_score.keys())

        filtered_db_results = self.store.retrieve_filtered_chunk_details(
            usearch_labels=usearch_labels,
            file_filter=file_filter,
            keyword_filter=keyword_filter,
        )

        final_results: List[SearchResult] = []
        for record in filtered_db_results:
            score = label_to_score.get(record["usearch_label"])
            if score is not None and score >= min_score:
                final_results.append(
                    SearchResult(
                        label=record["usearch_label"],
                        score=score,
                        file_path=record.get("file_path"),
                        chunk_text=record.get("chunk_text"),
                        start_char_offset=record.get("start_char_offset"),
                        end_char_offset=record.get("end_char_offset"),
                    )
                )

        final_results.sort(key=lambda r: r.score, reverse=True)
        return final_results