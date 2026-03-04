"""
LumenDB Python Client
=====================
A clean, idiomatic client for the LumenDB REST API.

Install dependencies:
    pip install requests
    pip install pandas          # optional — only needed for DataFrame helpers

Quick start::

    from lumendb_client import LumenDB

    with LumenDB("http://localhost:7070", api_key="secret") as db:
        db.create_collection("items", dim=384, metric="cosine")
        db.insert("items", [0.1, 0.2, ...], metadata={"label": "cat"})
        hits = db.search("items", query_vector, k=5)
        for h in hits:
            print(h.id, h.distance, h.metadata)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    import pandas as pd


__all__ = [
    "LumenDB",
    "Collection",
    "CollectionInfo",
    "SearchHit",
    "LumenDBError",
]


# ── Exceptions ────────────────────────────────────────────────────────────────


class LumenDBError(Exception):
    """Raised when the LumenDB server returns a non-2xx response."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}  (HTTP {status_code})")
        self.status_code = status_code
        self.code = code
        self.message = message


# ── Response models ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CollectionInfo:
    """Metadata returned by the server for a single collection."""

    name:   str
    dim:    int
    metric: str
    count:  int

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "CollectionInfo":
        return cls(
            name=data["name"],
            dim=data["dim"],
            metric=data["metric"],
            count=data["count"],
        )


@dataclass(frozen=True)
class SearchHit:
    """A single result returned by a KNN search."""

    id:       int
    distance: float
    metadata: Dict[str, Any]

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SearchHit":
        return cls(
            id=data["id"],
            distance=data["distance"],
            metadata=data.get("metadata") or {},
        )


# ── Collection handle ─────────────────────────────────────────────────────────


class Collection:
    """
    A scoped handle to a single LumenDB collection.

    Obtained via :meth:`LumenDB.collection`.  All methods delegate to the
    parent client so that connection management stays centralised::

        with LumenDB(...) as db:
            col = db.collection("items")
            col.insert([0.1, 0.2, 0.3], metadata={"tag": "a"})
            hits = col.search([0.1, 0.2, 0.3], k=5)
    """

    def __init__(self, client: "LumenDB", name: str) -> None:
        self._client = client
        self.name    = name

    # ── Queries ───────────────────────────────────────────────────────────────

    def info(self) -> CollectionInfo:
        """Return live stats (count, dim, metric) for this collection."""
        return self._client.get_collection(self.name)

    def insert(
        self,
        vector:   Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert one vector. Returns the assigned integer ID."""
        return self._client.insert(self.name, vector, metadata)

    def search(
        self,
        query: Sequence[float],
        k:     int = 10,
    ) -> List[SearchHit]:
        """Return the *k* approximate nearest neighbours of *query*."""
        return self._client.search(self.name, query, k)

    def get(self, id: int) -> Dict[str, Any]:
        """Fetch the metadata stored for vector *id*."""
        return self._client.get_vector(self.name, id)

    def delete(self) -> None:
        """Delete this collection from the server (irreversible)."""
        self._client.delete_collection(self.name)

    # ── Pandas helpers ────────────────────────────────────────────────────────

    def insert_dataframe(
        self,
        df:            "pd.DataFrame",
        vector_col:    str,
        metadata_cols: Optional[List[str]] = None,
        *,
        batch_size:    int  = 100,
        show_progress: bool = False,
    ) -> List[int]:
        """Bulk-insert a DataFrame. See :meth:`LumenDB.insert_dataframe`."""
        return self._client.insert_dataframe(
            self.name, df, vector_col, metadata_cols,
            batch_size=batch_size, show_progress=show_progress,
        )

    def search_to_dataframe(
        self,
        query: Sequence[float],
        k:     int = 10,
    ) -> "pd.DataFrame":
        """Search and return results as a tidy DataFrame."""
        return self._client.search_to_dataframe(self.name, query, k)

    def __repr__(self) -> str:
        return f"Collection(name={self.name!r}, client={self._client!r})"


# ── Main client ───────────────────────────────────────────────────────────────


class LumenDB:
    """
    Client for the LumenDB vector search engine.

    Parameters
    ----------
    base_url:
        Root URL of the server, e.g. ``"http://127.0.0.1:7070"``.
    api_key:
        Sent as the ``X-API-KEY`` header when provided.
    timeout:
        Per-request timeout in seconds (default 30).

    Usage as a context manager (recommended)::

        with LumenDB("http://localhost:7070", api_key="secret") as db:
            info = db.create_collection("embeddings", dim=1536, metric="cosine")
            db.insert("embeddings", my_vector, metadata={"source": "doc_42"})
            hits = db.search("embeddings", query_vector, k=10)

    Usage without a context manager::

        db = LumenDB("http://localhost:7070")
        try:
            ...
        finally:
            db.close()
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7070",
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        self._session  = requests.Session()
        if api_key:
            self._session.headers["X-API-KEY"] = api_key

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "LumenDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP session and free connections."""
        self._session.close()

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _raise_for_error(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        try:
            body = resp.json()
            raise LumenDBError(
                resp.status_code,
                body.get("code", "UNKNOWN"),
                body.get("error", resp.text),
            )
        except (ValueError, KeyError):
            raise LumenDBError(resp.status_code, "UNKNOWN", resp.text)

    def _get(self, path: str) -> Any:
        resp = self._session.get(self._url(path), timeout=self._timeout)
        self._raise_for_error(resp)
        return resp.json()

    def _post(self, path: str, body: Any) -> Any:
        resp = self._session.post(self._url(path), json=body, timeout=self._timeout)
        self._raise_for_error(resp)
        return resp.json() if resp.content else None

    def _delete(self, path: str) -> None:
        resp = self._session.delete(self._url(path), timeout=self._timeout)
        self._raise_for_error(resp)

    # ── Health ────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, str]:
        """Ping the server. Returns ``{"status": "ok", "version": "x.y.z"}``."""
        return self._get("/health")

    # ── Collections ───────────────────────────────────────────────────────────

    def create_collection(
        self,
        name:            str,
        dim:             int,
        metric:          str = "cosine",
        *,
        m:               int = 16,
        ef_construction: int = 200,
        ef_search:       int = 50,
    ) -> CollectionInfo:
        """
        Create a new collection.

        Parameters
        ----------
        name:
            Unique collection name (no ``/``, ``\\``, or ``.``).
        dim:
            Vector dimensionality. Every vector inserted must match this.
        metric:
            Distance metric — ``"cosine"``, ``"euclidean"``, or ``"dot_product"``.
        m:
            HNSW max edges per node (default 16). Higher → better recall, more RAM.
        ef_construction:
            HNSW build quality (default 200). Higher → slower insert, better graph.
        ef_search:
            HNSW query quality (default 50). Higher → slower query, better recall.
        """
        data = self._post("/v1/collections", {
            "name":            name,
            "dim":             dim,
            "metric":          metric,
            "m":               m,
            "ef_construction": ef_construction,
            "ef_search":       ef_search,
        })
        return CollectionInfo._from_dict(data)

    def get_collection(self, name: str) -> CollectionInfo:
        """Return live stats for *name*."""
        return CollectionInfo._from_dict(self._get(f"/v1/collections/{name}"))

    def delete_collection(self, name: str) -> None:
        """Permanently delete the collection *name* and all its vectors."""
        self._delete(f"/v1/collections/{name}")

    def collection(self, name: str) -> Collection:
        """
        Return a :class:`Collection` handle scoped to *name*.

        Useful when you are working with a single collection for an extended
        period — avoids repeating the collection name on every call::

            with db.collection("products") as col:
                col.insert(vec, metadata={"sku": "ABC123"})
                hits = col.search(query_vec, k=5)

        Note: ``Collection`` also supports use as a context manager (delegates
        to the parent ``LumenDB`` context).
        """
        return Collection(self, name)

    # ── Vectors ───────────────────────────────────────────────────────────────

    def insert(
        self,
        collection: str,
        vector:     Sequence[float],
        metadata:   Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Insert a single vector and return its assigned ID.

        Parameters
        ----------
        collection:
            Target collection name.
        vector:
            The embedding as a plain Python list or any ``Sequence[float]``.
        metadata:
            Arbitrary JSON-serialisable dict stored alongside the vector.
        """
        data = self._post(f"/v1/collections/{collection}/vectors", {
            "vector":   list(vector),
            "metadata": metadata or {},
        })
        return int(data["id"])

    def search(
        self,
        collection: str,
        query:      Sequence[float],
        k:          int = 10,
    ) -> List[SearchHit]:
        """
        Return the *k* approximate nearest neighbours of *query*.

        Results are sorted by ascending distance (most similar first).
        """
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        data = self._post(f"/v1/collections/{collection}/search", {
            "vector": list(query),
            "k":      k,
        })
        return [SearchHit._from_dict(h) for h in data["results"]]

    def get_vector(self, collection: str, id: int) -> Dict[str, Any]:
        """Fetch the metadata stored for a vector by its integer ID."""
        return self._get(f"/v1/collections/{collection}/vectors/{id}")

    # ── Pandas integration ────────────────────────────────────────────────────

    def insert_dataframe(
        self,
        collection:    str,
        df:            "pd.DataFrame",
        vector_col:    str,
        metadata_cols: Optional[List[str]] = None,
        *,
        batch_size:    int  = 100,
        show_progress: bool = False,
    ) -> List[int]:
        """
        Bulk-insert every row of a DataFrame into *collection*.

        Parameters
        ----------
        collection:
            Target collection name.
        df:
            DataFrame where each row represents one embedding to insert.
        vector_col:
            Name of the column whose values are ``list[float]`` embeddings.
            NumPy arrays and Pandas Series are accepted transparently.
        metadata_cols:
            Columns to store as JSON metadata alongside the vector.
            ``None`` (default) stores *all* columns except *vector_col*.
        batch_size:
            Number of rows processed per iteration.  Tune this based on
            embedding size and network latency.
        show_progress:
            Print a simple ``N/total inserted`` counter to stdout.

        Returns
        -------
        list[int]
            Server-assigned integer IDs in the same row order as the DataFrame.

        Example
        -------
        ::

            import pandas as pd
            from lumendb_client import LumenDB

            df = pd.DataFrame({
                "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "title":     ["hello", "world"],
                "score":     [0.9, 0.8],
            })

            with LumenDB("http://localhost:7070") as db:
                db.create_collection("docs", dim=3, metric="cosine")
                ids = db.insert_dataframe(
                    "docs", df, vector_col="embedding",
                    metadata_cols=["title", "score"],
                    show_progress=True,
                )
        """
        if metadata_cols is None:
            metadata_cols = [c for c in df.columns if c != vector_col]

        ids:   List[int] = []
        total: int       = len(df)

        for start in range(0, total, batch_size):
            chunk = df.iloc[start : start + batch_size]

            for _, row in chunk.iterrows():
                vector   = _to_float_list(row[vector_col])
                metadata = _sanitise_metadata(
                    {col: row[col] for col in metadata_cols}
                )
                ids.append(self.insert(collection, vector, metadata))

            if show_progress:
                done = min(start + batch_size, total)
                print(f"\r  Inserted {done}/{total}", end="", flush=True)

        if show_progress:
            print(f"\r  Inserted {total}/{total} — done.")

        return ids

    def search_to_dataframe(
        self,
        collection: str,
        query:      Sequence[float],
        k:          int = 10,
    ) -> "pd.DataFrame":
        """
        Search and return results as a tidy :class:`pandas.DataFrame`.

        Columns are always ``id`` and ``distance`` followed by one column per
        metadata key found in the results.

        Requires ``pandas`` to be installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for search_to_dataframe(). "
                "Install it with:  pip install pandas"
            ) from exc

        hits = self.search(collection, query, k)
        rows = [{"id": h.id, "distance": h.distance, **h.metadata} for h in hits]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"LumenDB(base_url={self._base_url!r})"




def _to_float_list(value: Any) -> List[float]:
    """Coerce a vector value (list, numpy array, pandas Series) to list[float]."""
    if hasattr(value, "tolist"):        # numpy ndarray or pandas Series
        return [float(x) for x in value.tolist()]
    return [float(x) for x in value]


def _sanitise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numpy/pandas scalar types to plain Python types so that the
    standard ``json`` module can serialise the metadata dict without errors.
    """
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out
