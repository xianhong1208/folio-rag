"""Whole-instance backup and restore (``main.py backup`` / ``main.py restore``).

One archive holds a ``pg_dump`` of Folio's database (every table, the per-folder
``data_*`` vector tables included), the files under ``storage/`` and a manifest that
records what the archive was made from (Folio version, migration revision, embedding
model and dimension, segmenter, folders, checksums). See docs/backup.md.
"""
