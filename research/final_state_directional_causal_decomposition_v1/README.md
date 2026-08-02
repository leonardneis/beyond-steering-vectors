# Final-State Directional Causal Decomposition v1

This study tests whether the replicated seed-2 behavioral necessity effect is
carried by the independently frozen teacher direction at the exact state read
by the language-model head, or by the complementary final-state displacement.

The design is deliberately narrow. It reuses the 72 frozen prompts, the seed-2
subliminal and neutral adapters, the k=20 top-ranked module set, and the 25
frozen norm-matched controls from the completed parent studies. It performs no
new training, selection, layer search, token search, or feature discovery.

- [Preregistration](PREREGISTRATION.md)
- [Prospective decision matrix](DECISION_MATRIX.md)
- [Results](RESULTS.md)
- Manifest: `configs/validation/cat_final_state_directional_causal_decomposition_v1.yaml`
- Output namespace: `results/research/qwen7b_cat_final_state_directional_causal_decomposition_v1/`

The study is not yet executed. Scientific interpretation is prohibited until
the complete artifact audit passes.
