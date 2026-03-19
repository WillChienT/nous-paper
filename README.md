# Nous: Extracting and Injecting Cognitive Priors for Diverse LLM Agents

As LLM agents proliferate in prediction markets and collective decision-making, they introduce a systemic risk: **cognitive monoculture**. Agents built on the same foundation models converge on homogeneous reasoning patterns, undermining the epistemic diversity that makes collective intelligence possible.

We argue this convergence is a structural consequence of training objectives that learn mean-field approximations of human cognition. The missing layer is **cognitive preprocessing** — the pre-reasoning information processing that shapes which signals an agent attends to, how it perceives risk, and how quickly it updates beliefs.

## What is Nous?

**Nous** is a framework for extracting structured cognitive profiles from human prediction market behavior and injecting them into LLM agents to restore behavioral diversity.

The framework consists of:

1. **Nous Schema** — An 8-dimensional cognitive profile organized in a Core-Shell-Membrane architecture:
   - **Core** (near-immutable): Risk Perception, Time Scale Preference, Cognitive Style
   - **Shell** (slowly evolving): Attention Allocation, Belief Update Inertia, Domain Confidence
   - **Membrane** (reactive): Independence Index, Loss Response

2. **Extraction Pipeline** — Infers cognitive parameters from prediction market behavior using Prospect Theory curve fitting, attention decay models, and belief update analysis

3. **Translation & Injection** — Converts parametric profiles into natural-language cognitive instructions and installs them as the foundational layer of an agent's prompt

## Key Results

- Nous-injected agents produce **judgment-level differentiation** (not merely stylistic variation) on 4/5 tasks across 3 models (Qwen3-32B, DeepSeek-R1-32B, Llama 3.1-8B)
- A second-generation continuous translator achieves **+16.3% mean diversity improvement** over threshold-based translation
- Automated extraction **matches handwritten persona baselines** on textual diversity while being fully automated
- **"Correct answer bias"**: LLM safety alignment overrides injected priors on tasks with socially consensual answers — a self-constraining feature, not a bug

## Paper

Read the full paper: **[main.pdf](main.pdf)**

## Building from Source

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Requires TeXLive with standard packages (amsmath, tikz, pgfplots, natbib, algorithm2e, booktabs).

## File Structure

```
main.tex           — Full paper (31 pages incl. appendix)
references.bib     — 47 BibTeX entries
figures/           — TikZ source files for all figures
main.pdf           — Compiled PDF
```

## Related Code

- Extraction engine, injection mechanism, and experiment code: [github.com/WillChienT](https://github.com/WillChienT) (coming soon)

## License

Paper: [CC BY-NC-ND 4.0](LICENSE)

## Citation

```bibtex
@article{qian2026nous,
  title={Nous: Extracting and Injecting Cognitive Priors for Diverse LLM Agents},
  author={Qian, Haowei},
  journal={Preprint},
  year={2026}
}
```

## Contact

Haowei Qian — haowei0509@gmail.com
