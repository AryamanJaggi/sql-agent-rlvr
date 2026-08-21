import pytest

from data.spider_loader import estimate_difficulty, load_spider


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
