# Nous: An Attempt to Extract and Inject the Cognition Behind Prediction-Market Behavior

As LLM agents proliferate in prediction markets and collective decision-making, they risk a **cognitive monoculture**: agents built on shared foundation models produce correlated forecasts — recent measurement finds the errors of independently-developed frontier models correlated at r ≈ 0.77. Nous asks whether human cognitive diversity can be recovered from real trading behavior and transferred to LLM agents through prompts.

**The answer, measured on real Polymarket data, is a dissociation between the two halves of that pipeline.**

## Central Finding

**Extraction works, partially.** Across 100 wallets, 8 of 14 schema parameters are temporally stable (split-half ICC ≥ 0.5 with bootstrap CI lower bound > 0.3; contrarian score reaches ICC ≈ 0.9); wallets are identifiable from their profiles well above chance (top-1 retrieval 17–22% vs. a 1% random baseline); and two of four pre-specified dimensions rank-correlate with future realized profit out-of-sample, though the correlations do not survive behavioral-confound controls.

**Prompt-level injection does not measurably transmit it.** On a semantic embedding metric, structured injection shows no significant advantage over a length-matched control on any model, and the small output diversity it induces neither reduces ensemble error correlation nor improves Brier score — a null that persists across exploratory checks on sampling temperature (0.0–1.0), a deliberately more-diverse synthetic profile population, and the model-uncertain question subset.

**The compression happens in the channel.** Measuring the prompts themselves shows the structure-to-narrative translator emits semantically near-uniform prompts whose spread does not increase when profile spread does. The profile is compressed before the model sees it — which motivates deeper, below-the-prompt injection (parameter-efficient fine-tuning, activation steering) as the next experiment.

We position Nous as work that **measures** the cognitive-monoculture problem and the limits of a prompt-level remedy, not one that solves it.

## Paper

Read the V2 paper: **[Nous_V2_An_Attempt_to_Extract_and_Inject.pdf](Nous_V2_An_Attempt_to_Extract_and_Inject.pdf)** (37 pages)

The earlier V1 preprint ([Nous_Cognitive_Priors_for_Diverse_LLM_Agents.pdf](Nous_Cognitive_Priors_for_Diverse_LLM_Agents.pdf), March 2026) is kept for provenance. It should be read as the architectural proposal: several of its headline results (the +16.3% translator diversity gain, parity with handwritten persona baselines, judgment-level differentiation counts) did not survive the stricter metric discipline of the V2 reanalysis and are revised or retracted in V2.

## Reproduction Artifacts

The [`artifacts/`](artifacts/) directory contains the release described in the paper's reproducibility section: evaluation and analysis code, frozen extracted profiles (pseudonymized W001–W100), all generated persona prompts, and complete model outputs for every experiment.

- All reported analyses are reproducible **from the extracted profiles onward**; see [`artifacts/REPRODUCE.md`](artifacts/REPRODUCE.md) for per-table commands.
- The production extraction parameters fitted to live user behavior, the `translator_v2.py` source, the extractor source, and raw wallet trade histories are **not** released, consistent with the paper's grey-box boundary.

## Citation

```bibtex
@article{qian2026nous,
  title={Nous: An Attempt to Extract and Inject the Cognition Behind Prediction-Market Behavior},
  author={Qian, Haowei},
  journal={Preprint},
  year={2026}
}
```

## License

Paper and artifacts: [CC BY-NC-ND 4.0](LICENSE)

## Contact

Haowei Qian — will@novasurge.ai
