"""Original scalar, typed-page and reservation models; no tensor execution."""

from __future__ import annotations

from dataclasses import dataclass, field


def eligible_boundaries(
    match: int, interval: int, mla: set[int], groups: tuple[set[int], ...]
) -> tuple[int, ...]:
    """Return ascending endpoints in a fixed, already compatible snapshot."""
    if match < 0 or interval <= 0 or not groups:
        raise ValueError("nonnegative match, positive interval and groups required")
    common = mla.intersection(*groups)
    return (0,) + tuple(
        sorted(b for b in common if 0 < b <= match and b % interval == 0)
    )


@dataclass(frozen=True)
class ScalarState:
    """Exact accumulator and auxiliary previous-input state."""

    total: int = 0
    previous: int = 0


def advance(
    state: ScalarState, inputs: tuple[int, ...]
) -> tuple[ScalarState, tuple[tuple[int, int], ...]]:
    """Outputs are the accumulator and two-tap sum after each input."""
    outputs = []
    for value in inputs:
        outputs.append((state.total + value, state.previous + value))
        state = ScalarState(state.total + value, value)
    return state, tuple(outputs)


@dataclass(frozen=True)
class Page:
    """MLA range is initialized coverage, never inferred from allocation size."""

    ident: str
    size: int
    kind: str
    boundary: int
    group: str = ""
    initialized: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.size <= 0 or self.boundary < 0 or self.kind not in {"MLA", "KDA"}:
            raise ValueError("invalid page")
        if self.initialized[0] < 0 or self.initialized[1] < self.initialized[0]:
            raise ValueError("invalid initialized range")


@dataclass
class Pool:
    """All methods are atomic owner operations in this sequential event model."""

    capacity: int
    pages: dict[str, Page] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)
    owned: dict[str, set[str]] = field(default_factory=dict)
    pins: dict[str, int] = field(default_factory=dict)
    writers: set[str] = field(default_factory=set)
    retained: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("negative capacity")

    @property
    def live(self) -> int:
        return sum(page.size for page in self.pages.values())

    @property
    def committed(self) -> int:
        return self.live + sum(self.reserved.values())

    def reserve(self, owner: str, amount: int) -> bool:
        if amount < 0 or owner in self.reserved or owner in self.owned:
            raise ValueError("invalid or repeated reservation")
        if self.committed + amount > self.capacity:
            return False
        self.reserved[owner] = amount
        self.owned[owner] = set()
        return True

    def materialize(self, owner: str, page: Page) -> None:
        if page.ident in self.pages or page.size > self.reserved.get(owner, -1):
            raise ValueError("duplicate allocation or insufficient reservation")
        self.reserved[owner] -= page.size
        self.pages[page.ident] = page
        self.owned[owner].add(page.ident)
        self.pins[page.ident] = 0

    def releasable(self, owner: str) -> bool:
        return owner not in self.retained and all(
            self.pins[p] == 0 and p not in self.writers
            for p in self.owned.get(owner, ())
        )

    def release(self, owner: str) -> bool:
        """Repeated completion is harmless; live readers/writers prevent release."""
        if not self.releasable(owner):
            return False
        for ident in self.owned.pop(owner, set()):
            del self.pages[ident]
            del self.pins[ident]
        self.reserved.pop(owner, None)
        return True


@dataclass(frozen=True)
class Job:
    """Finite admitted memory peak and number of work quanta."""

    ident: str
    peak: int
    quanta: int


@dataclass
class LocalScheduler:
    """Finite FIFO admission and equal-quantum cyclic service, without preemption."""

    pool: Pool
    waiting: list[Job] = field(default_factory=list)
    ready: list[str] = field(default_factory=list)
    remaining: dict[str, int] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    finished: set[str] = field(default_factory=set)

    def submit(self, job: Job) -> None:
        if (
            job.ident in self.seen
            or job.ident in self.pool.owned
            or job.peak < 0
            or job.quanta <= 0
        ):
            raise ValueError("invalid job or repeated identity")
        self.seen.add(job.ident)
        self.waiting.append(job)

    def admit(self) -> str:
        """One FIFO decision: empty, rejected, waiting or admitted."""
        if not self.waiting:
            return "empty"
        job = self.waiting[0]
        if job.peak > self.pool.capacity:
            self.waiting.pop(0)
            return "rejected"
        if not self.pool.reserve(job.ident, job.peak):
            return "waiting"
        self.waiting.pop(0)
        self.ready.append(job.ident)
        self.remaining[job.ident] = job.quanta
        return "admitted"

    def step(self) -> str | None:
        """One ready work unit; no elapsed-time or GPU model is implied."""
        if not self.ready:
            return None
        ident = self.ready.pop(0)
        self.remaining[ident] -= 1
        if self.remaining[ident]:
            self.ready.append(ident)
        return ident

    def finish(self, ident: str) -> bool:
        if ident in self.finished:
            return True
        if ident not in self.remaining:
            raise ValueError("request is not admitted")
        if self.remaining.get(ident, 0) != 0:
            raise ValueError("work has not completed")
        if not self.pool.release(ident):
            return False
        self.remaining.pop(ident, None)
        self.finished.add(ident)
        return True

    def cancel(self, ident: str) -> bool:
        """Stop future work; retire device ownership before final byte release."""
        if ident in self.finished:
            return True
        for index, job in enumerate(self.waiting):
            if job.ident == ident:
                self.waiting.pop(index)
                self.finished.add(ident)
                return True
        if ident not in self.remaining:
            raise ValueError("unknown request")
        self.ready = [item for item in self.ready if item != ident]
        self.remaining[ident] = 0
        return self.finish(ident)
