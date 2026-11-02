"""logging.max_size / backup_count actually rotate and prune log files."""

import os
import sys

from loguru import logger


def test_rotated_files_are_pruned_to_backup_count(tmp_path):
    from src.log import get_api_logger, setup_logger
    try:
        setup_logger(console_level="CRITICAL", log_base_dir=str(tmp_path),
                     rotation="1 KB", retention=2)
        log = get_api_logger()
        for i in range(300):
            log.info("x" * 60 + str(i))
        api_dir = tmp_path / "api"
        info_files = [f for f in os.listdir(api_dir) if f.startswith("info_")]
        assert 2 <= len(info_files) <= 3  # the live file + at most 2 rotated ones
    finally:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")


def test_non_positive_backup_count_means_keep_all(tmp_path):
    from src.log import setup_logger
    try:
        setup_logger(console_level="CRITICAL", log_base_dir=str(tmp_path), retention=0)
    finally:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
