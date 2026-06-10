# Prompt-space diversity: translator compression test

Real-10 vs mock-14 persona prompts; embedding = nomic-embed-text.

| Layer | Real mean (max) | Mock mean (max) | Mock/Real mean ratio |
|---|---|---|---|
| Input (5-dim std.) | 5.1541 (7.3157) | 5.3796 (10.6637) | 1.044 |
| Prompt (embedding cosine) | 0.016 (0.0328) | 0.0162 (0.0275) | 1.01 |
| Prompt (trigram Jaccard) | 0.284 (0.3978) | 0.3011 (0.4035) | 1.06 |

Within-set Spearman (input distance vs prompt embedding distance):
- real: rho = 0.375 (p = 0.0112)
- mock: rho = 0.359 (p = 0.0005)
- pooled: rho = 0.346 (p = 3.7e-05)

Permutation test (mock vs real mean pairwise prompt distance): diff = 0.0002, p = 0.9347

Reading: if the prompt-space mock/real ratio stays near (or below) the
input-space ratio and the input-prompt correlation is weak, the translator
is the compressing stage; if prompts spread but outputs did not, the model is.
