
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, BigInteger, Index, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
import uuid

from src.config.config_manager import Config

# Lazy init: do not create the engine at import time, so the application can load
# configuration (Config.set_config) before any DB connection is attempted.
_engine = None
Session = sessionmaker()
Base = declarative_base()


class Folder(Base):
    """Folder model."""
    __tablename__ = 'Folders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    file_count = Column(Integer, default=0)
    total_size = Column(BigInteger, default=0)
    # user_token stores the token that owns this folder (used for authorization filtering).
    # Note: not a FK, because tokens are managed by the remote Token Server.
    user_token = Column(String(256), nullable=True)
    vector_table_uuid = Column(PostgresUUID(as_uuid=True), default=uuid.uuid4, nullable=False, unique=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


    def __repr__(self):
        return f"<Folder(id={self.id}, name='{self.name}')>"


class File(Base):
    """Database model for file records."""
    __tablename__ = 'Files'

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    folder_id = Column(Integer, ForeignKey('Folders.id'), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(100))
    description = Column(Text)
    tags = Column(JSONB, nullable=False, default=list, server_default='[]')
    # sha256 computed at upload; compared against FileIndex.content_hash at index time to skip re-embedding when unchanged
    content_hash = Column(String(64), nullable=True)
    upload_time = Column(DateTime, default=func.now())
    updated_time = Column(DateTime, default=func.now(), onupdate=func.now())


    __table_args__ = (
        Index('idx_files_folder_id', 'folder_id'),
        Index('idx_files_upload_time', 'upload_time', postgresql_using='btree', postgresql_ops={'upload_time': 'DESC'}),
        Index('idx_files_folder_name', 'folder_id', 'file_name'),
    )

    def __repr__(self):
        return f"<FileRecord(id={self.id}, file_name='{self.file_name}', folder_id={self.folder_id}, size={self.file_size})>"


class FileIndex(Base):
    """Tracks which files have been indexed in LlamaIndex vector stores"""
    __tablename__ = 'FileIndices'

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_id = Column(PostgresUUID(as_uuid=True), ForeignKey('Files.id'), nullable=False, unique=True)
    folder_id = Column(Integer, ForeignKey('Folders.id'), nullable=False)

    # LlamaIndex document ID (used to delete from vector store)
    index_id = Column(String(255), nullable=False)

    # Vector store table name (format: {folder_id}_{vector_table_uuid})
    vector_store_table = Column(String(255), nullable=True)

    chunk_size = Column(Integer, default=256)
    chunk_overlap = Column(Integer, default=50)
    # The application layer always passes a model name; NULL = unknown, never a fabricated legacy default
    embedding_model = Column(String(100), nullable=True)
    num_chunks = Column(Integer, default=0)

    status = Column(String(50), default="indexed")  # indexed, failed, deleted
    error_message = Column(Text, nullable=True)

    # Snapshot of File.content_hash at index time; a match lets a later reindex short-circuit
    content_hash = Column(String(64), nullable=True)

    indexed_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


    __table_args__ = (
        Index('idx_fileindices_folder_id', 'folder_id'),
        Index('idx_fileindices_file_id', 'file_id'),
        Index('idx_fileindices_status', 'status'),
    )

    def __repr__(self):
        return f"<FileIndex(id={self.id}, file_id={self.file_id}, status='{self.status}')>"


class RuntimeSetting(Base):
    """Runtime setting overrides, layered on top of config.yaml at startup.

    key = dotted path (e.g. "rag.rerank.score_threshold"); value is JSONB wrapped as
    {"value": ...} because a bare scalar at the JSONB top level behaves inconsistently
    across drivers. config.yaml stays the factory default; this table holds live,
    individually resettable adjustments.
    """
    __tablename__ = 'RuntimeSettings'

    key = Column(String(255), primary_key=True)
    value = Column(JSONB, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    updated_by = Column(String(64), nullable=True)  # masked token prefix, for audit


class SettingsAudit(Base):
    """Append-only audit log of setting changes (RuntimeSetting keeps only the current
    value); records who changed what, to what, and when."""
    __tablename__ = 'SettingsAudit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), nullable=False)
    action = Column(String(16), nullable=False)  # set / reset
    old_value = Column(JSONB, nullable=True)
    new_value = Column(JSONB, nullable=True)
    changed_by = Column(String(64), nullable=True)  # masked token / "console"
    changed_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index('idx_settings_audit_at', 'changed_at'),)


class IndexJob(Base):
    """Persistence for background index jobs. Written through by IndexingJobManager so
    job state survives restarts.

    Timestamps are ISO strings (matching IndexJobState serialization) to avoid timezone drift.
    """
    __tablename__ = 'IndexJobs'

    job_id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    folder_id = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    current_index = Column(Integer, default=0)
    current_file_id = Column(String(255), nullable=True)
    current_file_name = Column(String(255), nullable=True)
    last_file_status = Column(String(64), nullable=True)
    last_message = Column(Text, nullable=True)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    skip_existing = Column(Boolean, default=True)
    started_at = Column(String(64), nullable=True)
    completed_at = Column(String(64), nullable=True)
    last_updated_at = Column(String(64), nullable=True)
    result_summary = Column(JSONB, nullable=True)
    scope_file_ids = Column(JSONB, nullable=False, default=list, server_default='[]')
    file_timings = Column(JSONB, nullable=False, default=list, server_default='[]')

    __table_args__ = (
        Index('idx_indexjobs_folder_id', 'folder_id'),
        Index('idx_indexjobs_status', 'status'),
        Index('idx_indexjobs_last_updated', 'last_updated_at'),
    )

    def __repr__(self):
        return f"<IndexJob(job_id={self.job_id}, folder_id={self.folder_id}, status='{self.status}')>"


class QueryLog(Base):
    """Retrieval query log feeding analytics (volume / popular / zero-result / latency).

    One best-effort row per retrieval (a logging failure never blocks the query). Only a
    masked token prefix is kept, never the full token; operational telemetry, not
    authorization data.
    """
    __tablename__ = 'QueryLogs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    folder_id = Column(Integer, nullable=True)
    query = Column(Text, nullable=False)
    result_count = Column(Integer, default=0)
    latency_ms = Column(Integer, nullable=True)
    token_prefix = Column(String(64), nullable=True)  # masked prefix
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_querylogs_folder', 'folder_id'),
        Index('idx_querylogs_created', 'created_at'),
    )


def _resolve_db_url(db_conf) -> str:
    """Return the full connection URL, injecting the config `database.port` when the URL
    omits one.

    create_engine ignores the separate port field, so a portless URL silently defaults
    to 5432 and a user who set the port would hit the wrong instance. Injecting it keeps
    all connection paths (sync / async / vector store) consistent.
    """
    from sqlalchemy import make_url
    url = make_url(db_conf.url)
    port = getattr(db_conf, "port", None)
    if url.port is None and port:
        url = url.set(port=int(port))
    return url.render_as_string(hide_password=False)


def get_engine():
    """Return the configured SQLAlchemy engine, creating it from Config if not yet built.

    Note: Config.set_config(...) must be called before this function to load the configuration.
    """
    global _engine
    if _engine is None:
        # Read the database config from the Pydantic model; raise if the model isn't loaded
        db_conf = Config.get_database_config()
        if not db_conf:
            raise RuntimeError("Database configuration is missing in Pydantic ConfigModel.")
        db_uri = _resolve_db_url(db_conf)
        # Default False: echo=True prints every SQL statement to the log — high volume
        # and possibly containing data values. Set echo: true in config to debug SQL.
        echo_flag = getattr(db_conf, 'echo', False)
        _engine = create_engine(
            db_uri,
            echo=echo_flag,
        )
        Session.configure(bind=_engine)
    return _engine


_async_engine = None


def get_async_engine():
    """Return the shared SQLAlchemy async engine (asyncpg driver), created lazily.

    Used on the query hot path for DB I/O that must not block the event loop. Reads the
    same config as get_engine() with the driver rewritten to postgresql+asyncpg://; its
    pool is counted independently from the sync engine.
    """
    global _async_engine
    if _async_engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine
        db_conf = Config.get_database_config()
        if not db_conf:
            raise RuntimeError("Database configuration is missing in Pydantic ConfigModel.")
        async_uri = _resolve_db_url(db_conf).replace(
            "postgresql://", "postgresql+asyncpg://", 1)
        _async_engine = create_async_engine(
            async_uri,
            echo=getattr(db_conf, 'echo', False),
            pool_size=getattr(db_conf, 'pool_size', 5),
            max_overflow=getattr(db_conf, 'max_overflow', 10),
            pool_recycle=getattr(db_conf, 'pool_recycle', 3600),
        )
    return _async_engine


def init_db(create_tables: bool = True):
    """Initialize the database connection, optionally creating tables.

    Args:
        create_tables: True → run Base.metadata.create_all (idempotent against an existing schema).

    Returns:
        The configured SQLAlchemy engine.
    """
    engine = get_engine()
    if create_tables:
        Base.metadata.create_all(engine)
    return engine
