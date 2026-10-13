"""Atomic original owner. GiB are integral fixture units, not measured bytes."""

from dataclasses import dataclass, field

from .reference import BUDGETS

GrantKey = tuple[str, int, int, str]


@dataclass(frozen=True)
class Contract:
    """Per-attempt work, request output ceiling, and absolute deadline in ms."""

    work: tuple[int, int, int] = (100, 4, 0)
    classes: tuple[str, ...] = ("L",)
    output_cap: int = 4
    deadline: int = 200

    def __post_init__(self) -> None:
        if (
            len(self.work) != 3
            or any(value < 0 for value in self.work)
            or not self.classes
            or any(cls not in BUDGETS for cls in self.classes)
            or self.output_cap < 0
        ):
            raise ValueError("invalid execution contract")


@dataclass
class Grant:
    """One immutable peak contract and its changing physical ownership."""

    peak: int
    context: str
    contract: Contract
    served: tuple[int, int, int] = (0, 0, 0)
    state: str = "prepared"
    reserved: int = 0
    live: dict[str, int] = field(default_factory=dict)
    materialized: dict[str, int] = field(default_factory=dict)


class Owner:
    """Methods are serialized; external sinks are assumed to enforce epoch."""

    def __init__(self, epoch: int = 5, capacity: int = 16, background: int = 4) -> None:
        if not 0 <= background <= capacity:
            raise ValueError("invalid initial pool")
        self.epoch = epoch
        self.capacity = capacity
        self.background = background
        self.grants: dict[GrantKey, Grant] = {}
        self.active: dict[str, GrantKey] = {}
        self.spent: dict[tuple[int, str, int], int] = {}
        self.quanta: dict[tuple[GrantKey, str], tuple[int, str, int, int]] = {}
        self.outputs: dict[tuple[str, int], str] = {}
        self.visible: dict[tuple[str, int], str] = {}
        self.latest_slot = -1
        self.request_limits: dict[str, tuple[int, int]] = {}

    def occupancy(self) -> tuple[int, int]:
        live = self.background + sum(sum(g.live.values()) for g in self.grants.values())
        reserved = sum(g.reserved for g in self.grants.values())
        return live, reserved

    def prepare(
        self,
        key: GrantKey,
        peak: int,
        context: str = "v7",
        contract: Contract | None = None,
    ) -> bool:
        if contract is None:
            contract = Contract()
        if peak < 0 or context != "v7":
            raise ValueError("invalid peak or context")
        if key in self.grants:
            grant = self.grants[key]
            if (grant.peak, grant.context, grant.contract) != (peak, context, contract):
                raise ValueError("conflicting duplicate")
            return grant.state != "released"
        limits = (contract.output_cap, contract.deadline)
        if key[0] in self.request_limits and self.request_limits[key[0]] != limits:
            raise ValueError("retry changed logical request limits")
        if key[2] != self.epoch:
            return False
        if sum(self.occupancy()) + peak > self.capacity:
            return False
        self.grants[key] = Grant(peak, context, contract, reserved=peak)
        self.request_limits[key[0]] = limits
        return True

    def commit(self, key: GrantKey) -> bool:
        grant = self.grants[key]
        if key[2] != self.epoch or grant.state not in ("prepared", "committed"):
            return False
        current = self.active.get(key[0])
        if current is not None and current != key:
            return False
        grant.state = "committed"
        self.active[key[0]] = key
        return True

    def authorized(self, key: GrantKey) -> bool:
        return (
            key[2] == self.epoch
            and self.grants[key].state == "committed"
            and self.active.get(key[0]) == key
        )

    def allocate(self, key: GrantKey, allocation: str, amount: int) -> bool:
        grant = self.grants[key]
        if allocation in grant.materialized:
            if grant.materialized[allocation] != amount:
                raise ValueError("conflicting allocation retry")
            return grant.state != "released"
        if not self.authorized(key):
            return False
        if amount < 0 or amount > grant.reserved:
            return False
        if any(allocation in g.materialized for g in self.grants.values()):
            raise ValueError("allocation ID belongs to another grant")
        grant.reserved -= amount
        grant.live[allocation] = amount
        grant.materialized[allocation] = amount
        return True

    def dispatch(
        self, key: GrantKey, quantum: str, slot: int, cls: str, axis: int, work: int
    ) -> bool:
        """Immediate bounded service within its named slot; no in-flight reset."""
        if cls not in BUDGETS or axis not in range(3) or slot < 0 or work <= 0:
            raise ValueError("invalid service quantum")
        if axis == 0 and work > 10:
            raise ValueError("prefill quantum exceeds ten")
        identity = (key, quantum)
        payload = (slot, cls, axis, work)
        if identity in self.quanta:
            if self.quanta[identity] != payload:
                raise ValueError("conflicting quantum retry")
            return True
        if slot < self.latest_slot:
            return False
        if not self.authorized(key):
            return False
        counter = (slot, cls, axis)
        grant = self.grants[key]
        if (
            cls not in grant.contract.classes
            or 10 * (slot + 1) >= grant.contract.deadline
            or grant.served[axis] + work > grant.contract.work[axis]
        ):
            return False
        if self.spent.get(counter, 0) + work > BUDGETS[cls][axis]:
            return False
        self.spent[counter] = self.spent.get(counter, 0) + work
        self.quanta[identity] = payload
        served = list(grant.served)
        served[axis] += work
        grant.served = (served[0], served[1], served[2])
        self.latest_slot = slot
        return True

    def retire(self, key: GrantKey) -> None:
        grant = self.grants[key]
        if grant.state != "released":
            grant.state = "retiring"
        if self.active.get(key[0]) == key:
            del self.active[key[0]]

    def release(self, key: GrantKey, *, quiescent: bool) -> bool:
        grant = self.grants[key]
        if grant.state == "released":
            return True
        if grant.state != "retiring" or not quiescent:
            return False
        grant.live.clear()
        grant.reserved = 0
        grant.state = "released"
        return True

    def advance_epoch(self, epoch: int) -> None:
        if epoch <= self.epoch:
            raise ValueError("epochs must increase")
        for key in self.grants:
            self.retire(key)
        self.epoch = epoch

    def resize(self, capacity: int) -> bool:
        if capacity < sum(self.occupancy()):
            return False
        self.capacity = capacity
        return True

    def publish(self, key: GrantKey, sequence: int, payload: str, *, now: int) -> bool:
        if sequence < 0 or not self.authorized(key):
            return False
        identity = (key[0], sequence)
        if identity in self.outputs:
            if self.outputs[identity] != payload:
                raise ValueError("conflicting committed output")
            return True
        contract = self.grants[key].contract
        if sequence >= contract.output_cap or now >= contract.deadline:
            return False
        if sequence and (key[0], sequence - 1) not in self.outputs:
            return False
        self.outputs[identity] = payload
        return True

    def deliver(self, request: str, sequence: int) -> str:
        """Stored payload retry with client deduplication; acknowledgement is external."""
        identity = (request, sequence)
        payload = self.outputs[identity]
        self.visible.setdefault(identity, payload)
        return payload
