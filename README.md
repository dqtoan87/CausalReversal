# CausalReversal

Code, manuscript sources and numerical results for the paper

**Causal Analysis of Learning Reversal under Selective Verification in Skin Cancer Classification**
Quang Toan Dao and Viet Anh Nguyen, Institute of Information Technology, Vietnam Academy of Science and Technology.

## Layout

| Path | Contents |
| --- | --- |
| `Code/` | Analysis scripts `vr*.py`, their drivers `run_*.sh`, the shared modules (`sv_common.py`, `vl_common.py`, `vl_models.py`) and the scripts that build the frozen inputs (`vl0_embed.py`, `vl1_identification.py`, `vl3_learning.py`, `sv2_bracket.py`) |
| `Result/` | JSON output of every script, including per-replicate bootstrap files in `vr19/`, `vr24/`, `vr30/`, `vr36/` and `vr39/`, and the frozen patient split `vl_split.json` |
| `paper/` | Manuscript and supplement sources (`paper_src.md`, `supplementary_src.md`), the builders that fill in every number from the JSON files, the built `paper.md` and `supplementary.md`, and the figures |
| `analysis_lock.md` | Analysis lock: primary, secondary and exploratory outcomes and the claims allowed for each |

Supplementary Table S48 maps every table, figure and number in the paper to the script and output that produce it.

## Data

The two data sets are public and are not redistributed here.

* ISIC-2024 (SLICE-3D), from the ISIC Archive. Place `train-metadata.csv` and the image file under `data_ISIC2024/`.
* PAD-UFES-20, from Mendeley Data.

Set `CAUSALREVERSAL_DATA` to the folder that holds both:

```bash
export CAUSALREVERSAL_DATA=/path/to/data
```

Model checkpoints, image embeddings (`Result/embeddings/`) and cached nuisance arrays (`Result/vl_nuisance*.npz`) are not included because of their size. `vl0_embed.py`, `vl1_identification.py` and `vl3_learning.py` recreate them. Keep `Result/vl_split.json` unchanged: rebuilding the split without retraining every model breaks the patient-disjoint property.

## Rebuilding the manuscript from the stored results

No training is needed for this step.

```bash
cd paper
python3 build_paper.py
python3 build_supplementary.py
python3 check_split.py paper.md supplementary.md
```

## Rerunning the analyses

```bash
pip install -r requirements.txt
cd Code
python3 -m pytest -q test_reversal_theory.py
```

Each script writes its JSON to `Result/`. The long bootstrap runs are split into ranges that can run in parallel and resume where they stopped; the `run_*.sh` drivers show the ranges used. The primary three-seed bootstrap of Table 1 needs `vr19_primary_bootstrap.py` and `vr39_seed_variance.py` over replicates 0 to 4,999, followed by `vr43_three_seed_primary.py`. A GPU is used for the learners when one is available.

## License

MIT, see `LICENSE`.
