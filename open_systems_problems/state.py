"""Finite identity and effect-boundary models; payload truth is an assumption."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Identity:
    """Required model, numerical, media and text-prefix identities."""

    weights: str = "v0"
    numerical: str = "n0"
    diagram: str = "d0"
    preprocessing: str = "p0"
    prefix: str = "common-1-12"


@dataclass(frozen=True)
class Package:
    """Two required checkpoint groups plus MLA and auxiliary readiness."""

    identity: Identity = Identity()
    ready_groups: tuple[frozenset[int], ...] = (
        frozenset({4, 8, 12}),
        frozenset({4, 8}),
    )
    mla_through: int = 12
    auxiliary_ready: bool = True


def joint_boundary(package: Package, target: Identity, through: int = 12) -> int | None:
    """Largest jointly ready positive boundary; None explicitly means rebuild."""
    if through < 0 or package.mla_through < 0:
        raise ValueError("negative boundary")
    if (
        package.identity != target
        or len(package.ready_groups) != 2
        or not package.auxiliary_ready
    ):
        return None
    common = set.intersection(*(set(group) for group in package.ready_groups))
    valid = [b for b in common if 0 < b <= min(through, package.mla_through)]
    return max(valid, default=None)


@dataclass
class EffectStore:
    """Atomic generation-check model; intentionally unsafe arm exposes stale DMA."""

    epoch: int = 7
    committed: int = 14
    payload: str = "prefix14"
    published: list[int] = field(default_factory=list)

    def takeover(self) -> None:
        self.epoch += 1

    def device_write(self, owner: int, payload: str, *, fenced: bool = True) -> bool:
        # The check and effect are one indivisible model transition.
        if fenced and owner != self.epoch:
            return False
        self.payload = payload
        return True

    def commit(self, owner: int, position: int, *, ready: bool) -> bool:
        if (
            owner != self.epoch
            or not ready
            or position != self.committed + 1
            or self.payload != f"prefix{position}"
        ):
            return False
        self.committed = position
        self.published.append(position)
        return True
