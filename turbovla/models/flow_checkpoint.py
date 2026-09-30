"""Architecture checks for V15 flow-state checkpoint compatibility."""

from __future__ import annotations

import re
from typing import Any, Mapping


_FLOW_PREFIX = "flow_action_policy."
_BLOCK_PATTERN = re.compile(r"^flow_action_policy\.flow\.bijection\.(\d+)\.")


def _state_dict(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"unsupported checkpoint payload: {type(payload)}")
    for key in ("model_state_dict", "model", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def _metadata(payload: Mapping[str, Any], top_level: str, flow_key: str) -> int | None:
    values = []
    if payload.get(top_level) is not None:
        values.append(int(payload[top_level]))
    flow_config = payload.get("flow_config")
    if isinstance(flow_config, Mapping) and flow_config.get(flow_key) is not None:
        values.append(int(flow_config[flow_key]))
    if len(set(values)) > 1:
        raise RuntimeError(f"checkpoint has inconsistent {top_level} metadata: {values}")
    return values[0] if values else None


def _encoding_metadata(payload: Mapping[str, Any]) -> str | None:
    values = []
    if payload.get("flow_state_encoding") is not None:
        values.append(str(payload["flow_state_encoding"]))
    flow_config = payload.get("flow_config")
    if isinstance(flow_config, Mapping) and flow_config.get("state_encoding") is not None:
        values.append(str(flow_config["state_encoding"]))
    if len(set(values)) > 1:
        raise RuntimeError(f"checkpoint has inconsistent flow_state_encoding metadata: {values}")
    return values[0] if values else None


def infer_flow_checkpoint_architecture(
    payload: Any, *, state_dim: int
) -> tuple[int, int, str] | None:
    """Infer ``(flow_state_dim, bijection_blocks, state_encoding)``.

    Metadata-free historical flow checkpoints are the original two-state-token,
    six-block route.  This intentionally does not infer the whole V15 model from
    bare weights; it only protects the flow route from partial loading.
    """
    raw = _state_dict(payload)
    source = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in raw.items()
    }
    has_flow = any(key.startswith(_FLOW_PREFIX) for key in source)
    has_projection = any(key.startswith("state_proj_module.") for key in source)
    has_state_encoder = any(key.startswith(_FLOW_PREFIX + "state_encoder.") for key in source)
    if not (has_flow or has_projection):
        return None
    if has_projection and has_state_encoder:
        raise RuntimeError("checkpoint mixes legacy StateProjection and direct-state parameters")

    declared_state = _metadata(payload, "flow_state_dim", "state_dim")
    declared_encoding = _encoding_metadata(payload)
    combined_route = declared_encoding == "state_tokens_zero_pad"
    inferred_state = (int(state_dim) if combined_route else 0) if has_projection else (int(state_dim) if has_state_encoder else None)
    if declared_state is not None and inferred_state is not None and declared_state != inferred_state:
        raise RuntimeError(
            "checkpoint flow_state_dim metadata conflicts with parameter structure: "
            f"metadata={declared_state}, inferred={inferred_state}"
        )
    route_state = declared_state if declared_state is not None else (inferred_state if inferred_state is not None else 0)
    if route_state not in (0, int(state_dim)):
        raise RuntimeError(
            f"checkpoint flow_state_dim={route_state} is incompatible with action state_dim={state_dim}"
        )

    indices = {int(match.group(1)) for key in source if (match := _BLOCK_PATTERN.match(key))}
    inferred_blocks = max(indices) + 1 if indices else None
    if indices and indices != set(range(inferred_blocks)):
        raise RuntimeError(f"checkpoint has non-contiguous flow bijection blocks: {sorted(indices)}")
    declared_blocks = _metadata(payload, "flow_bijection_blocks", "bijection_blocks")
    if declared_blocks is not None and inferred_blocks is not None and declared_blocks != inferred_blocks:
        raise RuntimeError(
            "checkpoint bijection_blocks metadata conflicts with parameter structure: "
            f"metadata={declared_blocks}, inferred={inferred_blocks}"
        )
    blocks = declared_blocks if declared_blocks is not None else (inferred_blocks if inferred_blocks is not None else 6)
    if blocks < 1:
        raise RuntimeError(f"checkpoint bijection_blocks must be positive, got {blocks}")
    inferred_encoding = ("state_tokens_zero_pad" if combined_route else "state_tokens") if has_projection else ("native_mlp" if has_state_encoder else None)
    if declared_encoding is not None and inferred_encoding is not None and declared_encoding != inferred_encoding:
        raise RuntimeError(
            "checkpoint flow_state_encoding metadata conflicts with parameter structure: "
            f"metadata={declared_encoding}, inferred={inferred_encoding}"
        )
    if route_state == 0:
        encoding = declared_encoding or "state_tokens"
        if encoding != "state_tokens":
            raise RuntimeError("legacy flow_state_dim=0 checkpoint must use state_tokens encoding")
    else:
        encoding = declared_encoding or inferred_encoding
        if encoding not in {"native_mlp", "zero_pad", "state_tokens_zero_pad"}:
            raise RuntimeError(
                "direct-state checkpoint has no state-encoding metadata and no native state_encoder; "
                "cannot distinguish zero_pad from an incomplete checkpoint"
            )
        if encoding == "state_tokens_zero_pad" and not has_projection:
            raise RuntimeError("state_tokens_zero_pad checkpoint is missing StateProjection parameters")
    return route_state, blocks, encoding


def assert_flow_checkpoint_compatible(payload: Any, model: Any) -> tuple[int, int, str] | None:
    """Reject flow structure mismatches before non-strict loading masks them."""
    if getattr(model, "action_head_type", None) != "flow_matching":
        return None
    architecture = infer_flow_checkpoint_architecture(payload, state_dim=int(model.state_dim))
    if architecture is None:
        return None
    source_state_dim, source_blocks, source_encoding = architecture
    target = (
        int(model.flow_state_dim),
        int(model.flow_bijection_blocks),
        str(model.flow_state_encoding),
    )
    if architecture != target:
        raise RuntimeError(
            "incompatible flow checkpoint architecture: "
            f"checkpoint uses flow_state_dim={source_state_dim}, state_encoding={source_encoding}, "
            f"bijection_blocks={source_blocks}; current model uses flow_state_dim={target[0]}, "
            f"state_encoding={target[2]}, bijection_blocks={target[1]}. "
            "Build the checkpoint's original architecture or start a new run; implicit migration is unsupported."
        )
    return architecture


def assert_flow_parameters_loaded(model: Any, missing: list[str], skipped: list[str] = ()) -> None:
    """Refuse a flow load that left route/block parameters randomly initialized."""
    if getattr(model, "action_head_type", None) != "flow_matching":
        return
    prefixes = ["flow_action_policy.flow.bijection."]
    if int(model.flow_state_dim) == 0 or str(model.flow_state_encoding) == "state_tokens_zero_pad":
        prefixes.append("state_proj_module.")
    elif str(model.flow_state_encoding) == "native_mlp":
        prefixes.append("flow_action_policy.state_encoder.")
    incomplete = [key for key in [*missing, *skipped] if key.startswith(tuple(prefixes))]
    if incomplete:
        raise RuntimeError(
            "checkpoint did not load all flow architecture parameters; refusing randomly initialized "
            f"state/coupling tensors: {incomplete[:10]}"
        )
