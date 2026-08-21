from unittest.mock import patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from data.spider_loader import _download_db, estimate_difficulty, load_spider


def test_easy_no_components():
    assert estimate_difficulty("SELECT name FROM singer") == "easy"


def test_medium_one_where():
    assert estimate_difficulty("SELECT name FROM singer WHERE age > 20") == "medium"


def test_medium_where_plus_order_by():
    sql = "SELECT name FROM singer WHERE age > 20 ORDER BY age"
    assert estimate_difficulty(sql) == "medium"


def test_hard_join_group_having():
    sql = (
        "SELECT s.name, COUNT(*) FROM singer s JOIN concert c ON s.id = c.singer_id "
        "GROUP BY s.name HAVING COUNT(*) > 1"
    )
    assert estimate_difficulty(sql) == "hard"


def test_extra_nested_and_set_op():
    sql = (
        "SELECT name FROM singer WHERE id IN (SELECT singer_id FROM concert) "
        "UNION SELECT name FROM singer WHERE age > 50 ORDER BY name"
    )
    assert estimate_difficulty(sql) == "extra"


def test_download_db_retries_on_transient_hf_error(tmp_path):
    fake_file = tmp_path / "concert_singer.sqlite"
    calls = {"n": 0}

    def flaky_download(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LocalEntryNotFoundError("rate limited")
        return str(fake_file)

    sleeps: list[float] = []
    with patch("data.spider_loader.hf_hub_download", side_effect=flaky_download):
        path = _download_db("train", "concert_singer", sleep=sleeps.append)

    assert calls["n"] == 3
    assert path == fake_file
    # Slept before the 2nd and 3rd attempts only, with growing backoff.
    assert sleeps == [2.0, 4.0]


def test_download_db_gives_up_after_max_retries(tmp_path):
    def always_fails(**kwargs):
        raise LocalEntryNotFoundError("still rate limited")

    with patch("data.spider_loader.hf_hub_download", side_effect=always_fails):
        with pytest.raises(LocalEntryNotFoundError):
            _download_db("train", "concert_singer", max_retries=3, sleep=lambda s: None)


@pytest.mark.integration
def test_load_spider_real_pipeline():
    """Hits the network (HF Hub). Proves the xlangai/spider <-> spider-corpus
    join and lazy .sqlite download actually work end to end. Skips if
    offline rather than failing the whole suite.
    """
    try:
        examples = load_spider(split="validation", limit=2)
    except OSError as e:
        pytest.skip(f"network unavailable: {e}")

    assert len(examples) == 2
    for ex in examples:
        assert ex.db_path.exists()
        assert ex.db_path.suffix == ".sqlite"
        assert ex.gold_sql
        assert ex.question
        assert ex.difficulty in ("easy", "medium", "hard", "extra")
