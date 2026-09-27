"""Hysteresis alert latch (stdlib only, zero repo imports).

Armed/latched semantics for the yield-drift defect-rate signal:

- ``update(rate)`` fires (returns/latches ``True``) when ``rate >= on_threshold``.
- Once latched, it clears (returns ``False``) only when ``rate < off_threshold``.
- Inside the band ``[off_threshold, on_threshold)`` the previous state holds,
  so noisy rates hovering around the threshold never flap.

Current call-site note (DOCS-ONLY, not wired): the drift trigger lives in
``src/backend/service.py::InspectionService.record_telemetry``
(``drift_triggered = window_len >= 10 and defect_ratio > 0.15``) with a mirror
in ``src/ui/app.py`` (``window_error_rate > 15.0``). Wiring was deliberately
NOT done: the trigger is embedded in DB-transactional service code with no
runnable import-clean seam (``service.py`` imports sqlalchemy at module level
and ``main.py``/``graph.py`` pull FastAPI/LangGraph), so a latch swap cannot
be proven import-clean under the stdlib-only test constraint. When a seam is
available, the 2-line wiring is::

    from .alert_hysteresis import HysteresisAlert
    _drift_latch = HysteresisAlert(on_threshold=0.15, off_threshold=0.12)
    # ... drift_triggered = window_len >= 10 and _drift_latch.update(defect_ratio)
"""


class HysteresisAlert:
    """Latched threshold alert with a dead band between off and on."""

    def __init__(self, on_threshold: float = 0.15, off_threshold: float = 0.12):
        on_threshold = float(on_threshold)
        off_threshold = float(off_threshold)
        if not (0.0 <= off_threshold < on_threshold <= 1.0 + 1e-12):
            # Allow on==1.0 edge but always require off < on.
            if not off_threshold < on_threshold:
                raise ValueError(
                    f"require off_threshold < on_threshold, got off={off_threshold} on={on_threshold}"
                )
            raise ValueError(
                f"thresholds must lie in [0, 1], got off={off_threshold} on={on_threshold}"
            )
        self._on = on_threshold
        self._off = off_threshold
        self._active = False

    @property
    def on_threshold(self) -> float:
        return self._on

    @property
    def off_threshold(self) -> float:
        return self._off

    @property
    def state(self) -> bool:
        """Current latched state (True = alert firing)."""
        return self._active

    def update(self, rate) -> bool:
        """Feed a new rate (0..1); returns the latched alert state."""
        rate = float(rate)
        if self._active:
            if rate < self._off:
                self._active = False
        else:
            if rate >= self._on:
                self._active = True
        return self._active

    def reset(self) -> None:
        """Force-clear the latch (e.g. line resumed / ticket resolved)."""
        self._active = False
