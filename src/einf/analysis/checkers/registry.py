from einf.analysis.checkers.base import CheckerAdapter
from einf.analysis.checkers.pyrefly import PyreflyAdapter
from einf.analysis.checkers.pyright import PyrightAdapter
from einf.analysis.checkers.ty import TyAdapter
from einf.analysis.checkers.zuban import ZubanAdapter

SUPPORTED_CHECKER_NAMES = ("pyright", "basedpyright", "zuban", "ty", "pyrefly")


def build_checker_adapters(
    checker_names: tuple[str, ...],
) -> tuple[CheckerAdapter, ...]:
    """Build checker adapters from CLI names in declared order.

    Repeated checker names are collapsed so duplicate configuration cannot
    amplify result size; each tool contributes at most one adapter.
    """
    adapters: list[CheckerAdapter] = []
    seen: set[str] = set()
    for checker_name in checker_names:
        if checker_name in seen:
            continue
        seen.add(checker_name)
        match checker_name:
            case "pyright":
                adapters.append(PyrightAdapter(name="pyright", executable="pyright"))
            case "basedpyright":
                adapters.append(
                    PyrightAdapter(name="basedpyright", executable="basedpyright")
                )
            case "pyrefly":
                adapters.append(PyreflyAdapter())
            case "ty":
                adapters.append(TyAdapter())
            case "zuban":
                adapters.append(ZubanAdapter())
            case _:
                raise ValueError(f"unsupported checker adapter: {checker_name}")
    return tuple(adapters)


__all__ = ["SUPPORTED_CHECKER_NAMES", "build_checker_adapters"]
