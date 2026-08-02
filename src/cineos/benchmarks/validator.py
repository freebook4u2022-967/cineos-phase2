"""Benchmark validation helpers."""

from .suite import BenchmarkSuite


def validate_suite(suite: BenchmarkSuite) -> tuple[str, ...]:
    errors = []
    if not suite.cases:
        errors.append("suite has no cases")
    if not suite.metric_definitions:
        errors.append("suite has no metric definitions")
    return tuple(errors)
