# Next steps (Module: Loader/Prep/Exec)
**Goal:** implement Unit 02/03/04 (deterministic loader → prep-spec → hardware exec), with minimal smoke tests.

- [ ] **Unit 02 — loader_layout.py**
  - Implement `resolve_register_shape`, `derive_rail_layout`, `build_loader_spec`, `loader_spec_to_json`.
  - Deterministic ordering; assert invariants; no silent fixes.
  - Add a quick smoke CLI to dump a LoaderSpec from a toy ψ.

- [ ] **Unit 03 — prep_circuit.py**
  - Implement `synthesize_init_circuit` and `simulate_counts` (multinomial from |amp|^2).
  - End-to-end `prep_run_bundle(nve_bundle, shots)`.

- [ ] **Unit 04 — hardware_loader.py**
  - Define `build_exec_spec(prep_spec, backend_target, shots, seed)` and `run_backend(exec_spec)` using free-tier runtime.
  - `verify_exec_hash` to bind spec↔receipt.

- [ ] **Repro fixtures**
  - Save `runs/*_backend_configuration.json` & `*_backend_properties.json` alongside receipts.

- [ ] **Sanity tests**
  - Tiny vectors (L=3→n=2) to verify padding, rail layout, and counts conservation.

