"""Book-authored defective aggregation for the analysis-audit repair task."""


def score(rows):
    """Return the required score for two (class, support, correct) rows."""
    if len(rows) != 2 or len({row[0] for row in rows}) != 2:
        raise ValueError("exactly two distinct classes are required")
    for label, support, correct in rows:
        if not isinstance(label, str) or not label:
            raise ValueError("class label must be nonempty text")
        if not isinstance(support, int) or isinstance(support, bool) or support <= 0:
            raise ValueError("support must be a positive integer")
        if not isinstance(correct, int) or isinstance(correct, bool):
            raise ValueError("correct must be an integer")
        if correct < 0 or correct > support:
            raise ValueError("correct must lie between zero and support")
    return sum(correct for _, support, correct in rows) / sum(
        support for _, support, correct in rows
    )
