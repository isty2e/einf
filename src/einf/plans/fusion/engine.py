from .registry import TUPLE_FUSION_RULES_BY_WINDOW
from .types import RuntimeStepFusion, RuntimeStepFusions, RuntimeSteps, TupleRunner


def discover_tuple_step_fusion(
    runtime_steps: RuntimeSteps,
    /,
    *,
    required_output_arity: int | None = None,
) -> RuntimeStepFusion[TupleRunner] | None:
    """Discover one best consecutive runtime-step fusion for tuple output."""
    total_steps = len(runtime_steps)
    for window_size in range(total_steps, 0, -1):
        fusion_rules = TUPLE_FUSION_RULES_BY_WINDOW.get(window_size, ())
        if not fusion_rules:
            continue
        for start in range(total_steps - window_size + 1):
            stop = start + window_size
            window = runtime_steps[start:stop]
            for fusion_rule in fusion_rules:
                runner = fusion_rule.build_runner(window)
                if runner is None:
                    continue
                input_arity = window[0].input_arity
                output_arity = window[-1].output_arity
                if (
                    required_output_arity is not None
                    and output_arity != required_output_arity
                ):
                    continue
                return RuntimeStepFusion(
                    name=fusion_rule.name,
                    start=start,
                    stop=stop,
                    input_arity=input_arity,
                    output_arity=output_arity,
                    runner=runner,
                )
    return None


def _discover_fusion_from_start(
    runtime_steps: RuntimeSteps,
    /,
    *,
    start: int,
) -> RuntimeStepFusion[TupleRunner] | None:
    """Return one best fusion that starts at one fixed step index."""
    total_steps = len(runtime_steps)
    for window_size in range(total_steps - start, 0, -1):
        fusion_rules = TUPLE_FUSION_RULES_BY_WINDOW.get(window_size, ())
        if not fusion_rules:
            continue
        stop = start + window_size
        window = runtime_steps[start:stop]
        for fusion_rule in fusion_rules:
            runner = fusion_rule.build_runner(window)
            if runner is None:
                continue
            return RuntimeStepFusion(
                name=fusion_rule.name,
                start=start,
                stop=stop,
                input_arity=window[0].input_arity,
                output_arity=window[-1].output_arity,
                runner=runner,
            )
    return None


def discover_tuple_step_fusions(runtime_steps: RuntimeSteps, /) -> RuntimeStepFusions:
    """Discover non-overlapping fused segments across full runtime-step chain."""
    total_steps = len(runtime_steps)
    if total_steps == 0:
        return ()

    fusions: list[RuntimeStepFusion[TupleRunner]] = []
    step_index = 0
    while step_index < total_steps:
        fusion = _discover_fusion_from_start(runtime_steps, start=step_index)
        if fusion is None:
            step_index += 1
            continue
        fusions.append(fusion)
        step_index = fusion.stop
    return tuple(fusions)


__all__ = [
    "discover_tuple_step_fusion",
    "discover_tuple_step_fusions",
]
