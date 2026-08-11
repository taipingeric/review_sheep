from __future__ import annotations

import threading
from contextvars import ContextVar

from review_sheep import Confidence, Finding, Lens, Location, Severity
from review_sheep.lenses import run_lenses_in_parallel


def test_lenses_run_concurrently_with_context_and_merge_in_stable_order() -> None:
    barrier = threading.Barrier(len(Lens), timeout=2)
    trace_session = ContextVar("trace_session", default="missing")
    trace_session.set("session-123")
    worker_threads: set[int] = set()
    lock = threading.Lock()

    def run_lens(lens: Lens) -> list[Finding]:
        with lock:
            worker_threads.add(threading.get_ident())
        barrier.wait()
        return [
            Finding(
                description=f"{lens.value}:{trace_session.get()}",
                location=Location(),
                severity=Severity.LOW,
                confidence=Confidence.CONFIRMED,
                lens=lens,
            )
        ]

    findings = run_lenses_in_parallel(run_lens)

    assert len(worker_threads) == len(Lens)
    assert [finding.lens for finding in findings] == list(Lens)
    assert all(finding.description.endswith(":session-123") for finding in findings)
