"""Neo4j Service — khởi tạo + quản lý kết nối Neo4j.

Cung cấp:
  - ``Neo4jClient`` class (full client, merged from ``app.neo4j``)
  - ``Neo4jService`` wrapper với singleton + lifecycle
  - ``get_db()`` singleton accessor
  - ``init_neo4j()`` factory

Usage:
    from app.services import Neo4jClient, Neo4jService

    # Direct client (backup scripts)
    client = Neo4jClient()
    rows = client.query("MATCH (t:Tire) RETURN t LIMIT 5")

    # Service wrapper (pipeline, health checks)
    service = Neo4jService()
    rows = service.query("MATCH (t:Tire) RETURN t LIMIT 5")
    service.ping()
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from app.config import NEO4J_URI, NEO4J_PASSWORD
from app.cypher.validator import CypherValidator
from app.response.normalizer import normalize_data

logger = logging.getLogger(__name__)

# ── Neo4j username ────────────────────────────────────────────────────────
_NEO4J_USER = os.environ.get("NEO4J_USER") or os.environ.get("NEO4J_USERNAME") or "neo4j"

# ── Retryable errors ──────────────────────────────────────────────────────
_RETRYABLE = (ServiceUnavailable, SessionExpired, TransientError, ConnectionError, TimeoutError)


# ═══════════════════════════════════════════════════════════════════════════
#  Neo4jClient
# ═══════════════════════════════════════════════════════════════════════════
class Neo4jClient:
    """Standalone Neo4j client với pooling, retry, metrics, tracing."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        max_retries: int | None = None,
        query_timeout: int | None = None,
        max_connection_lifetime: int | None = None,
        database: str | None = None,
    ):
        self._uri = uri or NEO4J_URI
        self._user = user or _NEO4J_USER
        self._password = password or NEO4J_PASSWORD
        self._database = database or os.environ.get("NEO4J_DATABASE") or "neo4j"
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("NEO4J_MAX_RETRIES", "2"))
        self.query_timeout = query_timeout if query_timeout is not None else int(os.environ.get("NEO4J_QUERY_TIMEOUT", "30"))
        self.max_connection_lifetime = max_connection_lifetime if max_connection_lifetime is not None else int(os.environ.get("NEO4J_MAX_CONN_LIFETIME", "3600"))
        self._driver = None
        self._open = False

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_lifetime=self.max_connection_lifetime,
            )
            self._open = True
        return self._driver

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self):
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                logger.exception("Error closing Neo4j driver")
            finally:
                self._driver = None
                self._open = False

    async def __aenter__(self) -> "Neo4jClient":
        _ = self.driver
        return self

    async def __aexit__(self, *exc_info) -> None:
        self.close()

    # ── Core query ───────────────────────────────────────────────────────

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int | None = None,
        validate: bool = False,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        if timeout is None:
            timeout = self.query_timeout
        db_name = database or self._database

        if validate:
            try:
                validator = CypherValidator()
                valid, reason = validator.validate(cypher, params=params)
                if not valid:
                    logger.warning("Refusing to execute invalid Cypher: %s", reason)
                    raise ValueError(f"Invalid Cypher: {reason}")
            except Exception:
                logger.exception("Validator error; refusing to execute Cypher")
                raise

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                with self.driver.session(database=db_name) as session:
                    result = session.run(cypher, params or {}, timeout=timeout)
                    data = [r.data() for r in result]

                data = normalize_data(data)
                latency = round(time.time() - start, 3)
                logger.info("[NEO4J] rows=%d latency=%.3fs db=%s", len(data), latency, db_name)
                self._record_metrics(latency, len(data), success=True)
                return data

            except _RETRYABLE as e:
                last_exc = e
                logger.warning("[NEO4J] retryable error (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, e)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                logger.error("[NEO4J] all %d attempts failed", self.max_retries + 1)
                raise
            except Exception as e:
                last_exc = e
                logger.exception("[NEO4J] non-retryable error (attempt %d)", attempt + 1)
                raise

        raise RuntimeError("Neo4j query failed after retries") from last_exc

    def run_in_tx(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        db_name = database or self._database
        with self.driver.session(database=db_name) as session:
            result = session.execute_read(lambda tx: list(tx.run(cypher, params or {})))
        return [r.data() for r in result]

    # ── Schema utilities ─────────────────────────────────────────────────

    def list_labels(self) -> list[str]:
        try:
            rows = self.query("CALL db.labels()")
            return sorted(r.get("label", "") for r in rows if r.get("label"))
        except Exception:
            logger.exception("Failed to list labels")
            return []

    def list_relationship_types(self) -> list[str]:
        try:
            rows = self.query("CALL db.relationshipTypes()")
            return sorted(r.get("relationshipType", "") for r in rows if r.get("relationshipType"))
        except Exception:
            logger.exception("Failed to list relationship types")
            return []

    def list_properties(self, label: str) -> list[str]:
        try:
            rows = self.query(f"MATCH (n:`{label}`) UNWIND keys(n) AS prop RETURN DISTINCT prop ORDER BY prop")
            return [r["prop"] for r in rows]
        except Exception:
            logger.exception("Failed to list properties for label %s", label)
            return []

    def check_indexes(self, required: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
        if required is None:
            required = [("Tire", "size"), ("Tire", "brand")]
        try:
            with self.driver.session() as session:
                result = session.run("CALL db.indexes()")
                rows = [dict(r) for r in result]
                text = "\n".join(str(r) for r in rows)
                missing = [(lbl, prop) for lbl, prop in required if prop not in text]
                auto_create = os.environ.get("NEO4J_AUTO_CREATE_INDEXES", "false").lower() in ("1", "true", "yes")
                if missing and auto_create:
                    for label, prop in missing:
                        try:
                            stmt = f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.`{prop}`)"
                            session.run(stmt)
                            logger.info("Created index %s.%s", label, prop)
                        except Exception:
                            logger.exception("Failed to create index for %s.%s", label, prop)
                return missing
        except Exception:
            logger.exception("Failed to fetch Neo4j indexes")
            return required

    def ping(self) -> bool:
        try:
            self.query("RETURN 1 AS ok", timeout=5)
            return True
        except Exception:
            logger.warning("[NEO4J] ping failed")
            return False

    def info(self) -> dict[str, Any]:
        try:
            version = self.query("CALL dbms.components() YIELD versions RETURN versions")[0]["versions"][0]
        except Exception:
            version = "unknown"
        return {
            "uri": self._uri,
            "database": self._database,
            "version": version,
            "labels": self.list_labels(),
            "relationship_types": self.list_relationship_types(),
        }

    def _record_metrics(self, latency: float | None, rows: int, *, success: bool):
        try:
            if success:
                if latency is not None and metrics.query_latency is not None:
                    metrics.query_latency.observe(latency)
                if metrics.neo4j_query_counter is not None:
                    metrics.neo4j_query_counter.labels(success="true").inc()
                if metrics.neo4j_rows is not None:
                    metrics.neo4j_rows.observe(rows)
            else:
                if metrics.neo4j_query_counter is not None:
                    metrics.neo4j_query_counter.labels(success="false").inc()
                if metrics.neo4j_query_failures is not None:
                    metrics.neo4j_query_failures.inc()
        except Exception:
            pass


# ── Module-level singleton ───────────────────────────────────────────────
_db_instance: Neo4jClient | None = None


def get_db() -> Neo4jClient:
    """Return singleton Neo4jClient (lazy init on first call)."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Neo4jClient()
    return _db_instance


# ═══════════════════════════════════════════════════════════════════════════
#  Neo4jService
# ═══════════════════════════════════════════════════════════════════════════
class Neo4jService:
    """Wrapper service quanh ``Neo4jClient`` với singleton + lifecycle."""

    def __init__(self, client: Neo4jClient | None = None):
        self.client = client or get_db()

    def query(self, cypher: str, params: dict[str, Any] | None = None, **kwargs) -> list[dict[str, Any]]:
        return self.client.query(cypher, params, **kwargs)

    def run_in_tx(self, cypher: str, params: dict[str, Any] | None = None, **kwargs) -> list[dict[str, Any]]:
        return self.client.run_in_tx(cypher, params, **kwargs)

    def list_labels(self) -> list[str]:
        return self.client.list_labels()

    def list_relationship_types(self) -> list[str]:
        return self.client.list_relationship_types()

    def check_indexes(self) -> list[tuple[str, str]]:
        return self.client.check_indexes()

    def ping(self) -> bool:
        return self.client.ping()

    def is_healthy(self) -> bool:
        try:
            return self.ping()
        except Exception:
            return False

    def get_driver(self):
        """Return Neo4j driver instance (dùng cho neo4j-graphrag)."""
        return self.client.driver

    def close(self):
        self.client.close()
        global _db_instance
        _db_instance = None


# ── Convenience factory ──────────────────────────────────────────────────
def init_neo4j() -> Neo4jService:
    """Factory: create + verify Neo4j connection."""
    logger.info("[SERVICES] Initialising Neo4jService...")
    service = Neo4jService()
    try:
        ok = service.ping()
        if ok:
            logger.info("[SERVICES] Neo4jService connected to %s", service.client._uri)
        else:
            logger.warning("[SERVICES] Neo4jService ping returned False")
    except Exception as e:
        logger.warning("[SERVICES] Neo4jService NOT available: %s", e)
    return service
