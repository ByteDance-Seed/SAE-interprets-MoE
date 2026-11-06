# Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization

Code and data to reproduce the experiments of the paper on Qwen3-30B-A3B (layers 4, 30 and 46).

The activation of an SAE latent on the output of an MoE block decomposes exactly over the
activated experts, because the SAE encoder is affine. Averaging the resulting contribution
fractions over a corpus and thresholding them gives a bipartite *dominance graph* between experts
and latents, and auto-interpreting the latents an expert dominates describes what the expert does.

**Links:** [Project page](https://bytedance-seed.github.io/SAE-interprets-MoE/) ·
[SAE weights](https://huggingface.co/ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE) · [Paper](https://arxiv.org/abs/2611.08089)

## Layout

```
common/            shared helpers: paths, SAE loading, explanation files, LLM client
contribution/      the contribution statistics every experiment starts from (Section 3)
experiments/
  rank_profile/       Section 4.1, Figure 2
  dominance_graph/    Section 4.2, appendix "Degree distributions of the dominance graph"
  dominance_overlap/  Section 4.3, Table 1, appendix "Pairwise overlap of dominant sets"
  wordclouds/         Section 5.1, Figures 3 and 4, appendix word clouds of all experts
  theme_check/        Section 5.2, Table 2
  expert_domains/     Section 5.3, Figure 5, appendix "Expert labels and domains"
data/              released explanations, labels, domains and theme judgements
docs/              the project page, served by GitHub Pages (see "Project page" below)
```

Every experiment directory has a `run.sh` that reproduces its results with the settings of the
paper, and every script documents its inputs, outputs and options in its docstring. All commands
below are run from this directory; results go to `outputs/`.

## Setup

```bash
pip install -r requirements.txt
```

The contribution statistics need the per-expert `ModuleList` of `Qwen3MoeSparseMoeBlock`, hence
`transformers<5`. Qwen3-30B-A3B in bf16 takes about 61 GB, and the fp32 SAE encoder of each traced
layer another 8 GB; tracing the three layers in one pass needs two 80 GB GPUs.

The LLM-based experiments (`theme_check`, `expert_domains`) use an OpenAI-compatible endpoint. Set
`OPENAI_API_KEY`, and `OPENAI_BASE_URL` for another compatible provider. The paper uses
`gpt-5.5-2026-04-24`; pass `--model` to change it.

### SAE weights

The TopK SAEs (2^20 latents, k = 100) are released at
[`ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE`](https://huggingface.co/ByteDance-Seed/Qwen3-30B-A3B-MoE-SAE)
(also reachable via the huggingface.cn mirror) and are downloaded on first use. To use local
copies, pass `--sae_paths` to `contribution/trace.py`.

The Hugging Face repo holds one directory per layer, `layer_{4,30,46}/`, with two files:

- `sae.safetensors`: four float32 tensors — `W_enc` of shape `[2048, 2^20]` (column `i` belongs
  to latent `i`), `b_enc` of shape `[2^20]`, `W_dec` of shape `[2^20, 2048]` (row `i` is the
  decoder vector of latent `i` and has unit L2 norm) and `b_dec` of shape `[2048]`.
- `config.json`: the layer, architecture, `k` and tensor shapes in machine-readable form.

For a raw MoE block output `y`, the SAE encodes as `TopK_100(ReLU(y @ W_enc + b_enc))` and
decodes as `acts @ W_dec + b_dec`. The encoder does not subtract `b_dec`, and no input
normalization is needed: the scaling used in training is folded into the weights. The model card
on Hugging Face gives the full training and evaluation details.

### Validation data

The statistics are computed on a ~1.5B-token validation set drawn from the Olmo 3 training corpus,
disjoint from the data the SAEs were trained on, mixed like the SAE training data:

| share | source |
|-------|--------|
| 5/8   | Dolma 3 pretraining data |
| 1/4   | Dolci-Think SFT (reasoning chains of thought) |
| 1/8   | Dolci-Instruct SFT (instruction following) |

Prepare it as parquet files with one document per row in a `text` column (chat data rendered to
plain text). Documents are truncated to 4096 tokens. Parts are named after the parquet file, so
keeping one file (or file prefix) per source lets you aggregate a single source later, which the
token word clouds need.

## Step 1: contribution statistics

```bash
bash contribution/run.sh /path/to/validation/*.parquet
```

`contribution/trace.py` hooks the MoE block of every traced layer and, during the forward pass,
encodes the layer output with the SAE and computes the contribution of every activated expert to
every active latent (Eq. 6 and 7), summed per (expert, latent). Nothing per token is written to
disk, and the forward pass stops after the deepest traced layer. Latents are counted as active
only when they are selected by TopK *and* have a positive activation. `contribution/aggregate.py`
sums the parts into `outputs/contribution/layer{L}/aggregated.safetensors`, from which the share
s(j, i) of Eq. 9 follows by normalizing over experts.

To spread the work over many GPUs, run `trace.py` on disjoint row ranges (`--idx_start`,
`--idx_end`) and call `aggregate.py` once they are all done.

## Experiments

| experiment | paper | needs | command |
|------------|-------|-------|---------|
| rank profile of gating vs. contribution | Section 4.1, Figure 2 | step 1 | `bash experiments/rank_profile/run.sh` |
| degree distributions of the dominance graph (τ = 0.2) | Section 4.2, appendix | step 1 | `bash experiments/dominance_graph/run.sh` |
| Jaccard overlap of dominant sets | Section 4.3, Table 1, appendix heat map | step 1 | `bash experiments/dominance_overlap/run.sh` |
| explanation and token word clouds | Section 5.1, Figures 3 and 4 | explanations; step 1 for token clouds | `bash experiments/wordclouds/run.sh` |
| theme membership of explanations | Section 5.2, Table 2 | released explanations, LLM | `bash experiments/theme_check/run.sh` |
| expert domains | Section 5.3, Figure 5, appendix table | released labels, LLM | `bash experiments/expert_domains/run.sh` |

Without the model or an API key, part of the results can be reproduced from the released data alone:

```bash
python experiments/wordclouds/explanation_wordcloud.py   # explanation word clouds of the released experts
python experiments/theme_check/table.py                  # Table 2
python experiments/expert_domains/domain_stats.py        # Figure 5, appendix table
```

We release the sampled latent explanations of experts 1, 50, 57 and 76. Experiments that need the
explanations of other experts are the explanation word clouds of those experts, labeling the
experts (`experiments/expert_domains/label_experts.py`) and the Random row of Table 2
(`WITH_RANDOM=1 bash experiments/theme_check/run.sh`). To run them, draw latents from the dominant
sets with `experiments/wordclouds/sample_latents.py` and explain them with the prompts in
`data/autointerp` (the explainer pipeline that gathers activating contexts and logit effects is not
included), writing one `e{expert}.jsonl` per expert in the format below. Since both the sample and
the explanations differ, the results will not match the paper's exactly.

`experiments/expert_domains/run.sh` groups the released labels into domains and computes the
statistics; set `LABELS` to group labels of your own instead. Labeling is a separate step, so it
never overwrites the released labels.

`domain_stats.py` additionally reports the overlap-vs-domain figures of Section 5.3 (19 of the 20
most overlapping pairs fall inside one domain) once `experiments/dominance_overlap` has been run.

The token word clouds of Figure 4 are drawn on the Dolma 3 part of the validation set only; set
`PRETRAIN_PATTERN` to the glob matching those parts, e.g.
`PRETRAIN_PATTERN="dolma3_*" bash experiments/wordclouds/run.sh`.

## Released data

```
data/
  layer30/
    explanations/e{1,50,57,76}.jsonl  auto-interp explanations of the latents sampled from these experts
    expert_labels.jsonl            the label GPT-5.5 wrote for each expert
    expert_groups.json             the partition of the experts into 11 domains
    theme_check/                   every yes/no judgement behind Table 2
  autointerp/
    system_prompt.txt              system prompt of the explainer
    user_prompt_template.txt       per-latent user prompt, {{fields}} filled per latent
```

- **Explanations.** For every expert of layer 30, up to 2000 latents were drawn uniformly from its
  dominant set (τ = 0.2) and explained by GPT-5.5 (appendix "Pipeline of auto-interpretability").
  The files of experts 1, 50, 57 and 76 are released. Each line is
  `{"feat": "30-F<latent id>", "explanation": ..., "confidence": ...}`. The file lists the exact
  sample, including latents whose explanation is `Uninterpretable` or `null`; the scripts skip
  those. `experiments/wordclouds/sample_latents.py` draws a fresh sample the same way.
- **Labels and domains.** Outputs of `label_experts.py` and `group_experts.py` used in the paper.
- **Theme judgements.** `e{expert}_{theme}.json` judges every interpretable explanation of the
  expert, and `e{expert}_{theme}_random.json` judges 2000 explanations drawn from all other
  experts with seed 0. In the random files, entries from experts other than the four above keep
  the latent id, expert and judgement but not the explanation text. Two random runs hold 1999
  judgements because one request failed.

## Project page

`docs/` holds the static project page, including the interactive explorer of all 128 experts of
layer 30. It is a prebuilt site that needs no build step: on GitHub, enable
*Settings → Pages → Deploy from a branch → `main` / `/docs`*, and the page is served at
<https://bytedance-seed.github.io/SAE-interprets-MoE/>. Page assets are produced inside the
project and copied into `docs/static/`; to change the page, edit `docs/index.html` and
`docs/static/` and push.

## Citation

```plain
@misc{zhuo2026saeinterpretsmoesparse,
      title={SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization}, 
      author={Anyuan Zhuo and Shaohua Zhang and Yuan Lin and Pinyan Lu},
      year={2026},
      eprint={2611.08089},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.08089}, 
}
```

## License

The code in this repository is released under the [Apache License 2.0](LICENSE).

Third-party models, data and services used by this work keep their own terms:

- **Qwen3-30B-A3B** ([`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)), the model
  under study, is released by the Qwen team under the Apache License 2.0.
- **Validation data.** The statistics are computed on data drawn from the Olmo 3 training corpus
  released by Ai2: Dolma 3, Dolci-Think-SFT and Dolci-Instruct-SFT, all made available under the
  [ODC Attribution License (ODC-BY) v1.0](https://opendatacommons.org/licenses/by/1-0/) and intended
  for research and educational use under Ai2's Responsible Use Guidelines. The released data
  contains information derived from these datasets. This repository does not redistribute the
  corpus itself.
- **GPT-5.5.** The latent explanations, expert labels, domains and theme judgements in `data/layer30`
  were generated by OpenAI's GPT-5.5 and are machine-generated, not human-written. Their use is
  subject to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/).
