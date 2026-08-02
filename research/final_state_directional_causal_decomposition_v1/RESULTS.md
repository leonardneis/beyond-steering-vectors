# Final-State Directional Causal Decomposition v1: results

Status: complete; scientific and local integrity audits passed.

## Audit and execution

- Successful HTCondor DAG: `179008` (`4/4` nodes complete,
  `DAG_STATUS_OK`).
- Execution commit: `6e22c2c34ae6f5f0544f4e24b7376f94af10e32f`.
- Manifest SHA-256:
  `9fee8b0ac5971b2ed440774802bdeaae6637cc611377de09b1965a0557eae1e5`.
- Required artifact inventory: `11/11`; completion markers, provenance,
  frozen inputs, finite arrays, directional identities, and checksums passed.
- The independent local re-audit reproduced every scientific checksum and
  audit field. Only the host-specific absolute manifest path differed.
- State-to-head maximum absolute error was `0.0` in both conditions.
- Maximum component/patch numerical error was
  `3.352508065290749e-05`, below the frozen `0.0001` tolerance.

The successful state artifacts have SHA-256 values
`8c4eb6bae0aae82fdc2de5c7101b789f639799bd8055c512833616b4cfcfd023`
(subliminal) and
`5dbaca53af260f73d2da7c17eefaf2b5bf6a63867a7f7a260227961210917e32`
(neutral). The audited aggregate has SHA-256
`eb635bb4a13facbcf7302a631a653deee1897c839755926306914a51d1c3a1bc`.

## Preregistered estimands

The frozen equivalence margin was
`epsilon = 0.04405641704135471`. Intervals are the preregistered stratified
prompt-bootstrap percentile intervals (`5,000` samples; seed `20260804`).

| Component | Mean | 90% interval | 95% interval |
|---|---:|---:|---:|
| `G_full` | -0.219307 | [-0.255859, -0.181399] | [-0.264310, -0.174300] |
| `G_T` | -0.011131 | [-0.014209, -0.008060] | [-0.014724, -0.007399] |
| `G_R` (primary) | -0.208176 | [-0.244484, -0.170747] | [-0.252188, -0.163691] |

The exact aggregate identity held:
`G_full = G_T + G_R`. The descriptive ratio `G_T/G_full` was `0.050756`;
under the preregistered evidence limits this is not interpreted as a literal
percentage mediated.

## Confirmatory gates

**FSD1 -- effect reconstruction: PASS.** `G_full` was negative, its 95%
interval excluded zero, and it differed from the frozen parent contrast
`-0.22028208520677353` by `0.0009753`, within the frozen reconstruction
tolerance.

**FSD2 -- teacher-axis sufficiency: NOT SUPPORTED.** The primary residual
90% interval was not contained in `[-epsilon,+epsilon]`; it was instead wholly
negative and far outside the equivalence region.

**FSD3 -- residual necessity: SUPPORTED.** The residual 95% interval was
wholly below `-epsilon`. The teacher component's 90% interval was wholly
inside the equivalence region, so it was small under the preregistered
practical threshold even though its interval did not include zero.

The frozen decision function returned:

```text
classification = residual_dominant
effect_reconstruction_pass = true
residual_equivalent = false
teacher_equivalent = true
residual_substantial_in_inherited_direction = true
```

Decision-matrix outcome: **GO -- residual dominant**. The prospectively fixed
next evidential step is Residual Direction Characterization before any
Teacher-State Rescue. This result does not itself define that future study.

## Descriptive secondary results

The pooled conclusion was not homogeneous in magnitude across the three fixed
prompt families:

| Prompt family | `G_full` | `G_T` | `G_R` |
|---|---:|---:|---:|
| direct preference | -0.425614 | -0.015163 | -0.410451 |
| identity affinity | -0.226910 | -0.014998 | -0.211912 |
| hypothetical choice | -0.005397 | -0.003233 | -0.002164 |

These family estimates are descriptive and do not alter the pooled primary
decision. In the paired set ranking, the top set exceeded `0/25` controls for
the full and residual components, and `2/25` for the teacher component, using
the stored rank convention where a larger value counts as a win.

The fixed Cat--Lion readout compatibility with the unit teacher direction was
`u^T t = 0.045276`.

## Interpretation within the preregistered evidence boundary

### Observed

The inherited causal parameter-ablation contrast was reconstructed at the
final LM-head input. Almost all of its output-relevant displacement, in the
preregistered practical sense, lay in the orthogonal complement of the single
frozen teacher direction. The teacher-aligned component was precise but small;
the complementary component was large, negative, and carried the inherited
effect direction.

### Supported conclusion

The one-dimensional frozen teacher axis is insufficient to explain the direct
final-state Cat--Lion necessity effect for seed 2, k=20. The previously
observed teacher-aligned activation necessity and the behavioral decision
effect therefore need not be implemented through the same scalar direction at
the final readout bottleneck. A residual-dominant final-state implementation is
the strongest preregistered classification.

### Not established

The result does not show that the complement is one feature, one circuit, or
"non-teacher information." It does not exclude distributed upstream teacher
computation, nonlinear or multidimensional teacher representations, or a path
that converges onto another output-relevant direction. It does not localize
where the residual contribution is formed, establish natural reachability of
component patches, generalize beyond seed 2/model/trait/k, or establish
multi-token behavioral mediation.

## Execution history before the successful run

Three pre-outcome attempts terminated fail-closed before producing scientific
state or aggregate artifacts: an absent Docker `USER` variable, a
shape-dependent low-precision LM-head reconstruction path, and notification
container/runtime-mount issues. Their corrections changed only execution and
integrity machinery; no hypothesis, endpoint, prompt, intervention, epsilon,
bootstrap rule, or decision rule changed. No study outcome was inspected
before DAG `179008` completed and its audit passed.
