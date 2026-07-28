# Lazy Triton Import for Qwen3-VL MoE

## Problem

Importing `cosmos_framework.model.generator.mot.unified_mot` eagerly imports
the Qwen3-VL MoE implementation. Its `moe.py` module immediately imports
`moe_kernels.py`, which requires Triton. This prevents Cosmos3-Edge's
Nemotron Dense model from loading in an Ascend environment even though that
model never uses the Qwen3-VL MoE grouped-matrix-multiplication path.

## Design

Remove the module-level import of `TOKEN_GROUP_ALIGN_SIZE_M` and
`_generate_permute_indices` from `moe.py`. Import those two symbols inside
`Qwen3VLMoeTextExpertsGroupedMm.forward()` immediately before they are used.

This keeps the existing grouped-MM implementation and behavior unchanged
when it is selected, while allowing the module and all non-grouped-MM model
paths to load without Triton. If a caller explicitly selects grouped-MM on a
system without Triton, the existing `ModuleNotFoundError` occurs at the point
where that Triton-dependent implementation is actually executed.

No fallback from grouped-MM to the naive implementation is added because a
silent backend change could alter performance and hide configuration errors.
No dependency versions or lock files are changed.

## Testing

Add a regression test that runs an isolated Python import with imports of
`triton` and `triton.language` deliberately blocked. The test imports
`unified_mot` and verifies that the import succeeds, proving that the
Nemotron/Dense path does not require Triton.

Run the new focused test first without the production change to confirm the
regression test fails for the expected missing-Triton reason. Then apply the
minimal production change and rerun the focused test. Finally run the
relevant MoE tests and lint the changed files.

## Scope

Only the Qwen3-VL MoE Triton import timing and its regression coverage are in
scope. Triton-Ascend support, grouped-MM kernel porting, dependency changes,
and unrelated refactoring are excluded.
