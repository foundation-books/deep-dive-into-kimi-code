"""Bounded admission and membership transitions in an atomic finite model."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pending:
    """A logical request waiting for capacity under a fixed absolute deadline."""

    request: str
    arrival: int
    peak: int
    cap: int = 4
    attempt: int = 1
    retry_at: int = 0


class AdmissionQueue:
    """FIFO pending work; terminal outcomes remain visible outside the queue."""

    def __init__(self, limit: int = 16) -> None:
        if limit < 1:
            raise ValueError("positive queue limit required")
        self.limit = limit
        self.pending: list[Pending] = []
        self.outcomes: dict[str, str] = {}
        self.admitted: dict[str, Pending] = {}
        self.retry_receipts: dict[Pending, str] = {}

    def offer(self, item: Pending, now: int) -> str:
        if item.request in self.outcomes:
            return self.outcomes[item.request]
        existing = next((p for p in self.pending if p.request == item.request), None)
        if existing is not None:
            if existing != item:
                raise ValueError("conflicting pending request")
            return "deferred"
        if item.peak < 0 or item.cap < 0 or item.attempt < 1:
            raise ValueError("invalid request contract")
        if item.peak > 12 or item.cap > 4 or item.attempt > 3:
            self.outcomes[item.request] = "rejected"
        elif now >= item.arrival + 200:
            self.outcomes[item.request] = "expired"
        elif len(self.pending) >= self.limit:
            self.outcomes[item.request] = "rejected"
        else:
            self.pending.append(item)
            self.pending.sort(key=lambda p: (p.arrival, p.request))
            return "deferred"
        return self.outcomes[item.request]

    def next_ready(self, now: int, free: int) -> Pending | None:
        for item in tuple(self.pending):
            if now >= item.arrival + 200:
                self.pending.remove(item)
                self.outcomes[item.request] = "expired"
        if not self.pending:
            return None
        item = self.pending[0]
        if now < max(item.retry_at, item.arrival) or item.peak > free:
            return None
        self.pending.pop(0)
        self.outcomes[item.request] = "admitted"
        self.admitted[item.request] = item
        return item

    def retry(self, item: Pending, now: int, old_fenced: bool) -> str:
        if item in self.retry_receipts:
            return self.retry_receipts[item]
        if self.admitted.get(item.request) != item:
            return "obsolete_attempt"
        if not old_fenced:
            return "waiting_authority"
        if item.attempt >= 3 or now >= item.arrival + 200:
            self.outcomes[item.request] = "failed"
            self.retry_receipts[item] = "failed"
            return "failed"
        if self.outcomes.get(item.request) != "admitted":
            raise ValueError("retry requires an admitted attempt")
        del self.outcomes[item.request]
        replacement = Pending(
            item.request,
            item.arrival,
            item.peak,
            item.cap,
            item.attempt + 1,
            now + 10 * item.attempt,
        )
        result = self.offer(replacement, now)
        self.retry_receipts[item] = result
        return result


@dataclass
class Cluster:
    """Readiness gates and retained owner obligations for one physical domain."""

    state: str = "requested"
    context: str = "v7"
    owners: set[str] = field(default_factory=set)


class Membership:
    """Version publication changes new assignment eligibility, not live owners."""

    def __init__(self) -> None:
        self.version = 1
        self.clusters: dict[str, Cluster] = {}

    def request(self, name: str) -> None:
        if name in self.clusters:
            raise ValueError("cluster already known")
        self.clusters[name] = Cluster()

    def install(self, name: str, context: str) -> None:
        cluster = self.clusters[name]
        if cluster.state != "requested":
            raise ValueError("installation requires requested state")
        cluster.context = context
        cluster.state = "installed"

    def activate(self, name: str, policy_ready: bool) -> bool:
        cluster = self.clusters[name]
        if cluster.state != "installed" or cluster.context != "v7" or not policy_ready:
            return False
        cluster.state = "active"
        self.version += 1
        return True

    def assign(self, name: str, request: str) -> bool:
        cluster = self.clusters[name]
        if cluster.state != "active":
            return False
        cluster.owners.add(request)
        return True

    def drain(self, name: str) -> None:
        cluster = self.clusters[name]
        if cluster.state != "active":
            raise ValueError("only active clusters can begin draining")
        cluster.state = "draining"
        self.version += 1

    def finish_owner(self, name: str, request: str, quiescent: bool) -> bool:
        if not quiescent:
            return False
        self.clusters[name].owners.discard(request)
        return True

    def handoff(
        self, source: str, target: str, request: str, verified: bool, quiescent: bool
    ) -> bool:
        old, new = self.clusters[source], self.clusters[target]
        if (
            request not in old.owners
            or not verified
            or not quiescent
            or new.state != "active"
        ):
            return False
        old.owners.remove(request)
        new.owners.add(request)
        return True

    def retire(self, name: str) -> bool:
        cluster = self.clusters[name]
        if cluster.state != "draining" or cluster.owners:
            return False
        cluster.state = "retired"
        self.version += 1
        return True
