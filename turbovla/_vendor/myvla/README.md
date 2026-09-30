# Vendored MyVLA subset

This directory contains the two Apache-2.0 source files required by the latest
TurboVLA flow-matching head:

- `fluxvla/models/blocks/cross_attention_dit.py`
- `fluxvla/models/heads/flow_matching_7d_head.py`

They were copied from the local MyVLA checkout on 2026-08-17. TurboVLA loads
them through `turbovla.models.components.myvla_action_head` without importing
the complete MyVLA dependency tree. See `third_party/licenses/Apache-2.0.txt`.
