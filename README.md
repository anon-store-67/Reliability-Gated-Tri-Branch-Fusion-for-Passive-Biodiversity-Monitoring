# Code Guide: `main.py`

This guide describes the supplied Python script: data preparation, model training, recording-grouped evaluation, fusion comparators, and diagnostic exports. The script executes sequentially at module level and shares intermediate arrays between stages. Importing it starts execution; it has no command-line argument parser or separate inference entry point.

## 1. Environment

| Item | Requirement |
|---|---|
| Platform | Kaggle/Linux environment; a CUDA GPU is recommended for training. `DEVICE` selects CUDA when available, otherwise CPU. |
| Python packages | `torch`, compatible `torchaudio`, `soundfile`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `timm`; `IPython` for the display wrapper below |
| Data | BirdCLEF+ 2026 competition files at `/kaggle/input/competitions/birdclef-2026` |
| Expected data files | `train.csv`, `taxonomy.csv`, `train_soundscapes_labels.csv`, `train_audio/`, `train_soundscapes/` |
| Outputs | `/kaggle/working/` (results, OOF arrays) and `/kaggle/working/novel_crg_cache/` (feature and checkpoint caches) |

No external pretrained weights are loaded. The source and temporal networks are initialized locally; the `timm` baselines explicitly use `pretrained=False`. Existing local focal-training checkpoints are reused when present.

The initial diagnostics print library versions and inspect installed `librosa` and `tensorflow`, but neither package is used by the model pipeline. Package versions are not pinned in the script.

## 2. How to Run

1. Attach the competition data in a Kaggle environment and enable a GPU for training.
2. Place `main.py` in the current working directory.
3. Execute the complete script in one process. It calls `display(...)` without importing it, so supply that function explicitly:

```python
import runpy
from IPython.display import display

runpy.run_path(
	"main.py",
	init_globals={"display": display},
	run_name="__main__",
)
```

Running `python main.py` in a standard interpreter without supplying `display` will fail at its first display call. No stages are selected or skipped by command-line flags.

For another filesystem layout, adapt `DATA_ROOT`, `CACHE_DIR`, the explicit `/kaggle/working/` export paths, and the diagnostics that enumerate `/kaggle/input`. The script uses multiprocessing data loaders with two workers and is intended for the Kaggle/Linux execution layout.

Main settings:

| Constant | Value | Meaning |
|---|---:|---|
| `SEED` | 2026 | Focal sampling, folds, data order, initialization |
| `SAMPLE_RATE` | 16000 | Audio resampling rate |
| `WINDOW_SECONDS` | 5 | Window length |
| `NUM_CLASSES` | 234 | Target classes |
| `NUM_FOLDS` | 5 | Outer recording-grouped folds |
| `FOCAL_PER_CLASS` | 12 | Maximum focal clips sampled per class |
| `SOURCE_EPOCHS` | 60 | Maximum source-CNN epochs (early stopping, patience 8) |
| `HYBRID_EPOCHS` | 30 | Temporal-branch epochs |
| `BASELINE_SOURCE_EPOCHS` | 7 | Focal-training epochs for architecture baselines |
| `BASELINE_FINETUNE_EPOCHS` | 20 | Soundscape fine-tuning epochs per outer fold |
| `KAPPA_GRID` | 2, 4, 8 | Support-shrinkage sweep |
| `GAMMA_GRID` | 2, 5, 8 | Reliability softmax-scale sweep |

## 3. Execution Stages

Stages below follow script order. Out-of-fold (OOF) predictions retain the row order of `windows_df`.

### 3.1 Data inspection
Lists the dataset files, parses `train_soundscapes_labels.csv`, and derives `site` and `recording_hour` from each filename. Prints library versions and any model artifacts found on disk.

### 3.2 Label construction
- Merges duplicate label rows into one record per (file, window).
- Keeps recordings with exactly 12 unique labeled windows (`windows_df`). Other recordings are excluded and counted.
- Builds the multi-hot target matrix `targets` (windows x 234) and the boolean `active_mask`.

### 3.3 Feature extraction
- `read_fixed_clip` reads a 5-s clip, converts to mono, resamples to 16 kHz, and pads or truncates to a fixed length. Focal crop positions use a filename hash; soundscape crops use annotated start times.
- Log-mel: 64 bins, `n_fft=1024`, `hop_length=640`, 80 to 7,800 Hz, per-clip standardization.
- Focal clips are sampled by `FOCAL_PER_CLASS` with a fixed seed. Both focal and soundscape features are cached as CPU float16 tensors.

### 3.4 Acoustic source branch (`AcousticSourceModel`)
- Three-block CNN (24, 48, 96 channels) giving a 128-D embedding and 234 logits.
- Trained only on focal audio. Classes with at least five sampled clips contribute approximately 20% to focal validation; smaller classes remain in training only.
- AdamW uses learning rate 0.002 and weight decay 0.0002; training uses cross-entropy with label smoothing 0.03. Focal validation loss selects the checkpoint.
- Evaluation produces fixed source logits and embeddings for every soundscape window; later temporal training does not update the source model.
- `source_coverage` marks classes with focal audio and controls acoustic-branch masking in the tri-branch gate.

### 3.5 Folds and ecological prior
- `make_balanced_group_folds` assigns whole recordings to folds. A fold never splits a recording.
- `fold_safe_prior` builds global, site, hour, and site-hour occurrence tables from training rows only, with shrinkage toward the parent level.
- `macro_auc` averages ROC-AUC over classes containing both labels. `classwise_auc` returns NaN for classes without both labels.

### 3.6 Temporal branch (`TemporalHybridModel`)
- Projects embeddings and source logits, adds site, hour, and window-position embeddings, then applies a BiGRU and 4-head attention with a bounded residual correction.
- `train_hybrid_fold` trains it on one set of recordings and predicts another.
- Training uses AdamW at 0.0015, weight decay 0.002, batches of up to 12 recordings, weighted binary cross-entropy, and a 0.015 source-logit distillation penalty.

### 3.7 Early gate variants
The initial outer-fold loop produces source, prior, source-plus-prior, temporal, and pilot binary-gate arrays. The pilot `reliability_gate` uses in-sample temporal training predictions. A separate `nested_reliability_gate` then estimates a binary gate from inner OOF temporal predictions. These arrays are distinct from the subsequent tri-branch system.

### 3.8 Nested tri-branch reliability gate
`nested_tri_branch_weights` does the following inside each outer training set:
1. Splits the outer training recordings into 3 inner grouped folds.
2. Produces inner out-of-fold predictions for the source, prior, and temporal branches.
3. Computes classwise inner ROC-AUC per branch. An undefined AUC is set to 0.5.
4. Shrinks it toward 0.5 by positive-file support, `n / (n + kappa)` with `kappa = 4`.
5. Masks the source branch for classes without focal audio.
6. Applies a softmax with `temperature = 5` to get classwise branch weights. This argument is the multiplicative reliability scale called `gamma` in the later helper.

At the outer fold, each branch's scores are rank-normalized across that fold's validation windows and combined with these weights. `rank_normalize` averages tied ranks and divides by the number of rows plus one. Outer validation labels are not used to fit these weights.

### 3.9 Statistics and rank-normalized comparisons
- `cluster_bootstrap_auc` computes single-system intervals from 500 recording-cluster bootstrap samples by default.
- `paired_cluster_bootstrap_delta` computes paired intervals from 250 samples, using identical sampled recordings for both systems.
- `foldwise_rank_scores` applies the same per-fold rank normalization to each branch. Their equal-weight average forms `uniform_tri_oof`.
- The script does not include the notebook's historical-score or final-array integrity checks.

### 3.10 Acoustic architecture baselines
MobileNetV3-Small (`mobilenetv3_small_050`), EfficientNet-B0, and ResNet-18 from `timm`, trained under the same protocol:
- one input channel, the same 64-bin log-mel input, and random initialization;
- 7 epochs on the same focal subset, then 20 epochs fine-tuned inside each outer training fold;
- no site, hour, prior, or temporal input.

### 3.11 Additional analyses
`compute_inner_branch_arrays` retrains inner temporal models once per outer fold and shares these arrays among the following comparators:

| Analysis | Implementation |
|---|---|
| Prior + temporal average | Equal average of `ranked_prior_oof` and `ranked_hybrid_oof` |
| Global AUC-weighted fusion | `global_weighted_fusion_weights` produces one three-branch weight vector per outer fold, shared by all classes |
| Best branch per class | `best_branch_per_class` selects maximum shrunk inner AUC, falling back to the prior below four positive files; this helper does not apply the source-coverage mask |
| Per-class logistic stacking | `fit_logistic_stack` fits balanced logistic regression on inner ranks when both labels have at least three examples; other classes use the ranked prior |
| Greedy ensemble selection | `ensemble_selection_weights` selects global branch proportions for up to 30 rounds |
| Mask comparison | `tri_branch_weights_from_inner` supports `hard` and `soft` modes only |
| Reliability grid | Nine `kappa`/`gamma` combinations; exports the mean and standard deviation of outer-fold AUCs, not pooled OOF grid AUC |
| Undefined inner AUC | Counts NaNs per branch and outer fold across all 234 targets, including globally absent classes |

The primary hard-mask gate uses inner seed offset 500; the additional pass uses offset 900. Consequently, the exported hard-versus-soft comparison changes both the mask and the inner predictions. The script does not implement no-mask, no-shrinkage, or repeated grouped-split experiments.

### 3.12 Leave-site-out evaluation
Each site is held out in turn. The code calls `train_hybrid_fold` on the remaining recordings and estimates gate weights through `nested_tri_branch_weights`. Inner validation remains recording-grouped rather than site-grouped.

The held-out site's embedding is not explicitly zeroed or mapped to a reserved unknown-site index. Missing site priors fall back to the global estimate, and missing site-hour pairs fall back to the average of the site and hour estimates. Only gate and prior scores are exported for this evaluation. Pooled scores are printed over evaluated rows; no leave-site-out bootstrap summary is generated.

### 3.13 CPU timing and computation count
`measure_cpu_latency` times two CPU neural forwards: the source CNN on one 5-s window and a freshly initialized temporal model on twelve windows. It uses five warm-up passes and 50 measured passes. The printed recording-level latency is `12 * source_median_ms + temporal_median_ms`; it excludes decoding, resampling, log-mel extraction, prior construction, ranking, and fusion.

Thread counts are not pinned, so the console's one-thread throughput wording is not enforced by the code. `count_flops` uses PyTorch `FlopCounterMode` when available. On failure, it prints a warning and returns twice the parameter count; that fallback is not a measured FLOP count. The timing helper moves `source_model` itself to CPU.

### 3.14 Final diagnostics
The final block prints class coverage lost by excluding partial recordings and exports single-system recording-bootstrap intervals for the eight displayed systems. It does not generate figure files or a partial-recording predictive-sensitivity experiment.

## 4. Output Files

Written to `/kaggle/working/`:

| File | Content |
|---|---|
| `novel_crg_results.csv` | Final raw-score branch and gate AUC table; overwrites an earlier interval table |
| `novel_crg_paired_deltas.csv` | Gate versus branch paired differences |
| `novel_crg_ranked_results.csv`, `novel_crg_ranked_paired_deltas.csv` | Fold-wise rank-normalized comparison |
| `novel_architecture_baselines.csv`, `novel_architecture_paired_deltas.csv` | Acoustic baselines and gate-minus-baseline differences |
| `novel_crg_oof_predictions.npz` | Raw branch scores, nested binary and tri-branch scores, fold IDs, targets, and gate weights; overwrites an earlier binary-only archive |
| `novel_crg_final_oof_predictions.npz` | Ranked branches, uniform fusion, tri-branch fusion, fold IDs, targets, and tri-branch weights |
| `novel_architecture_oof_predictions.npz` | Ranked acoustic-baseline predictions, proposed scores, fold IDs, and targets |
| `novel_crg_manifest.json` | Run configuration summary |
| `reviewer_response_fusion_baselines.csv` | Six added fusion variants, including the soft-mask gate |
| `reviewer_response_paired_deltas.csv` | Variants versus the original tri-branch gate and ranked prior |
| `reviewer_response_kappa_gamma_grid.csv` | `kappa`, `gamma`, `mean_outer_auc`, and `std_outer_auc` |
| `reviewer_response_masking_ablation.csv` | Original hard-mask versus additional-pass soft-mask results |
| `reviewer_response_undefined_auc_counts.csv` | `fold`, `source`, `prior`, and `hybrid` total undefined-AUC counts |
| `reviewer_response_leave_site_out.csv`, `reviewer_response_leave_site_out_oof.npz` | Per-site gate/prior AUCs and their OOF arrays |
| `reviewer_response_cpu_latency_flops.csv` | Two neural-forward timing rows with computation-count outputs |
| `reviewer_response_fig3_error_bars.csv` | Single-system bootstrap intervals; no figure is generated |
| `reviewer_response_oof_predictions.npz` | Added comparator arrays, soft-mask gate, best-branch choices, targets, and fold IDs |

## 5. Cache Behavior

- Feature tensors, the source focal checkpoint, and architecture focal checkpoints are loaded from `CACHE_DIR` when present.
- Temporal outer/inner models and acoustic outer-fold fine-tuning are rerun. Saved acoustic OOF archives are not loaded to skip training.
- Cache names do not encode every preprocessing or training setting. Use a separate cache directory or remove the affected cache entries when changing those settings or the dataset.
- Result exports use fixed filenames and overwrite previous exports.

## 6. Reproducibility Notes

- Soundscape validation groups by recording filename. Prior tables and temporal models exclude the corresponding validation recordings during fitting.
- The source CNN is selected using focal-only validation. Nested gate weights use inner predictions from the outer training recordings; the earlier pilot binary gate instead uses in-sample temporal predictions.
- Seeds are set for sampling and training, but deterministic PyTorch algorithms are not enabled and `torch.backends.cudnn.benchmark = True`. Exact numeric reproduction is not guaranteed.
- Rank normalization uses the evaluated fold/site batch and therefore depends on its composition.
- Bootstrap samples retain whole recordings. Intervals describe recording-resampling variation, not retraining or model-selection uncertainty.
- This guide documents the supplied script, not additional analyses that may exist in a separate notebook version.
