"""Per-client privacy budget allocation rules (pluggable).

n_i is the number of *training patients* of client i (the unit of privacy is the patient, D2).

* ``uniform``     eps_i = eps
* ``adaptive``    eps_i = eps * (a + (1 - a) * n_i / n_max)       (FedGuard default, a = 0.55)
* ``inverse``     eps_i = eps * (a + (1 - a) * n_min / n_i)       (ablation: small sites get more budget)
* ``equal_noise`` every client gets the noise multiplier that the *uniform* rule would give the largest
                  client; the eps_i that results is computed by the accountant (ablation). Returned here
                  as None, meaning "determined by the shared noise multiplier".
"""

from __future__ import annotations

from collections.abc import Callable

RULES: dict[str, Callable[[float, dict[str, int], float], dict[str, float | None]]] = {}


def rule(name: str):
    def deco(fn):
        RULES[name] = fn
        return fn

    return deco


@rule("uniform")
def uniform(eps: float, n: dict[str, int], a: float = 0.55) -> dict[str, float | None]:
    return {c: eps for c in n}


@rule("adaptive")
def adaptive(eps: float, n: dict[str, int], a: float = 0.55) -> dict[str, float | None]:
    n_max = max(n.values())
    return {c: eps * (a + (1 - a) * ni / n_max) for c, ni in n.items()}


@rule("inverse")
def inverse(eps: float, n: dict[str, int], a: float = 0.55) -> dict[str, float | None]:
    n_min = min(n.values())
    return {c: eps * (a + (1 - a) * n_min / ni) for c, ni in n.items()}


@rule("equal_noise")
def equal_noise(eps: float, n: dict[str, int], a: float = 0.55) -> dict[str, float | None]:
    return {c: None for c in n}


def allocate(
    rule_name: str, eps: float, n_patients: dict[str, int], a: float = 0.55
) -> dict[str, float | None]:
    """Target total epsilon per client under ``rule_name``."""
    if rule_name not in RULES:
        raise ValueError(f"unknown budget rule {rule_name!r}; choose from {sorted(RULES)}")
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    return RULES[rule_name](eps, n_patients, a)
