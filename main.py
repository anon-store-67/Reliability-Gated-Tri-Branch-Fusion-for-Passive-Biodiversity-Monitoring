from pathlib import Path

import os

import pandas as pd



DATA_ROOT = Path("/kaggle/input/competitions/birdclef-2026")

assert DATA_ROOT.exists(), f"Competition data not found: {DATA_ROOT}"



print(f"Dataset root: {DATA_ROOT}")

print("\nTop-level structure:")

for path in sorted(DATA_ROOT.iterdir()):

    if path.is_dir():

        files = [item for item in path.rglob("*") if item.is_file()]

        total_gib = sum(item.stat().st_size for item in files) / 1024**3

        suffix_counts = pd.Series([item.suffix.lower() or "<none>" for item in files]).value_counts()

        suffix_summary = ", ".join(f"{suffix}: {count}" for suffix, count in suffix_counts.head(6).items())

        print(f"  [DIR] {path.name:<28} files={len(files):>6} size={total_gib:>7.2f} GiB  ({suffix_summary})")

    else:

        print(f"  [FILE] {path.name:<27} size={path.stat().st_size / 1024**2:>8.2f} MiB")



csv_files = sorted(DATA_ROOT.rglob("*.csv"))

print(f"\nCSV files ({len(csv_files)}):")

for csv_path in csv_files:

    frame = pd.read_csv(csv_path)

    print(f"  {csv_path.relative_to(DATA_ROOT)}: shape={frame.shape}")

    print(f"    columns={frame.columns.tolist()}")

    print(frame.head(2).to_string(index=False, max_cols=12))



print("\nRepresentative files (maximum 5 per top-level directory):")

for directory in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):

    samples = [path.relative_to(DATA_ROOT) for path in directory.rglob("*") if path.is_file()][:5]

    print(f"  {directory.name}: {[str(path) for path in samples]}")

import importlib.util

import json

import platform

from collections import Counter



labels_df = pd.read_csv(DATA_ROOT / "train_soundscapes_labels.csv")

labels_df["label_list"] = labels_df["primary_label"].fillna("").astype(str).str.split(";")

active_labels = sorted({label for labels in labels_df["label_list"] for label in labels if label})

labeled_files = sorted(labels_df["filename"].unique())



print("Labeled soundscape summary")

print(f"  rows={len(labels_df)} files={len(labeled_files)} active_labels={len(active_labels)}")

print(f"  rows/file: min={labels_df.groupby('filename').size().min()}, median={labels_df.groupby('filename').size().median()}, max={labels_df.groupby('filename').size().max()}")

label_counts = Counter(label for labels in labels_df["label_list"] for label in labels if label)

print(f"  positives/class: min={min(label_counts.values())}, median={pd.Series(label_counts).median()}, max={max(label_counts.values())}")

print(f"  rare classes (<5 positive windows)={sum(count < 5 for count in label_counts.values())}")



missing_audio = [name for name in labeled_files if not (DATA_ROOT / "train_soundscapes" / name).exists()]

print(f"  missing labeled audio files={len(missing_audio)}")



print("\nAttached /kaggle/input roots:")

for path in sorted(Path("/kaggle/input").iterdir()):

    print(f"  {path}")



print("\nRuntime libraries:")

for package in ["torch", "torchaudio", "librosa", "soundfile", "sklearn", "timm", "tensorflow"]:

    spec = importlib.util.find_spec(package)

    if spec is None:

        print(f"  {package}: not installed")

    else:

        module = __import__(package)

        print(f"  {package}: {getattr(module, '__version__', 'installed')}")

print(f"  Python: {platform.python_version()}")



model_candidates = []

for root in [Path("/kaggle/input"), Path.home() / ".cache"]:

    if root.exists():

        for pattern in ("*.pt", "*.pth", "*.onnx", "saved_model.pb"):

            model_candidates.extend(root.rglob(pattern))

print(f"\nModel artifacts found: {len(model_candidates)}")

for path in model_candidates[:50]:

    print(f"  {path}")

import re

import soundfile as sf



def clock_to_seconds(value):

    hours, minutes, seconds = (int(part) for part in str(value).split(":"))

    return hours * 3600 + minutes * 60 + seconds



labels_df["start_sec"] = labels_df["start"].map(clock_to_seconds)

labels_df["end_sec"] = labels_df["end"].map(clock_to_seconds)

labels_df["site"] = labels_df["filename"].str.extract(r"_(S\d+)_", expand=False)

labels_df["recording_hour"] = labels_df["filename"].str.extract(r"_(\d{6})\.ogg$", expand=False).str[:2].astype(int)



print("Labeled-window conventions")

print(labels_df[["filename", "start_sec", "end_sec", "site", "recording_hour"]].head(15).to_string(index=False))

print(f"sites={sorted(labels_df['site'].unique())}")

print(f"recording hours={sorted(labels_df['recording_hour'].unique())}")

print(f"window durations={sorted((labels_df['end_sec'] - labels_df['start_sec']).unique())}")

print(f"maximum labeled end={labels_df['end_sec'].max()} seconds")



print("\nRepresentative audio metadata")

for filename in labeled_files[:5]:

    info = sf.info(DATA_ROOT / "train_soundscapes" / filename)

    print(f"  {filename}: sr={info.samplerate}, channels={info.channels}, duration={info.duration:.1f}s")



datasets_root = Path("/kaggle/input/datasets")

print("\nAttached datasets (two levels):")

for owner in sorted(datasets_root.iterdir()) if datasets_root.exists() else []:

    children = sorted(owner.iterdir()) if owner.is_dir() else []

    print(f"  {owner.name}: {[child.name for child in children[:20]]}")

import hashlib

import random

import time



import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchaudio

from sklearn.metrics import roc_auc_score



SEED = 2026

SAMPLE_RATE = 16_000

WINDOW_SECONDS = 5

NUM_CLASSES = 234

NUM_FOLDS = 5

FOCAL_PER_CLASS = 12

SOURCE_EPOCHS = 60

HYBRID_EPOCHS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True



taxonomy_df = pd.read_csv(DATA_ROOT / "taxonomy.csv")

class_names = taxonomy_df["primary_label"].astype(str).tolist()

class_to_index = {label: index for index, label in enumerate(class_names)}

assert len(class_names) == NUM_CLASSES



raw_labels = pd.read_csv(DATA_ROOT / "train_soundscapes_labels.csv")

raw_labels["start_sec"] = raw_labels["start"].map(clock_to_seconds)

raw_labels["end_sec"] = raw_labels["end"].map(clock_to_seconds)



def union_label_strings(values):

    labels = set()

    for value in values.dropna().astype(str):

        labels.update(label for label in value.split(";") if label)

    return sorted(labels)



all_windows_df = (

    raw_labels.groupby(["filename", "start_sec", "end_sec"], as_index=False)

    .agg(label_list=("primary_label", union_label_strings))

    .sort_values(["filename", "start_sec"])

    .reset_index(drop=True)

)

unique_file_sizes = all_windows_df.groupby("filename").size()

complete_files = unique_file_sizes[unique_file_sizes == 12].index

windows_df = all_windows_df[all_windows_df["filename"].isin(complete_files)].copy().reset_index(drop=True)

windows_df["site"] = windows_df["filename"].str.extract(r"_(S\d+)_", expand=False)

windows_df["hour"] = windows_df["filename"].str.extract(r"_(\d{6})\.ogg$", expand=False).str[:2].astype(int)

windows_df["site_index"] = pd.Categorical(windows_df["site"]).codes



targets = np.zeros((len(windows_df), NUM_CLASSES), dtype=np.float32)

for row_index, labels in enumerate(windows_df["label_list"]):

    for label in labels:

        targets[row_index, class_to_index[label]] = 1.0



duplicate_rows_removed = len(raw_labels) - len(all_windows_df)

file_sizes = windows_df.groupby("filename").size()

active_mask = targets.sum(axis=0) > 0

print(f"Device: {DEVICE}")

print(f"Unique labeled windows={len(all_windows_df)}, duplicate rows merged={duplicate_rows_removed}")

print(f"Complete validation windows={len(windows_df)}, files={windows_df['filename'].nunique()}")

print(f"Excluded partial files={all_windows_df['filename'].nunique() - windows_df['filename'].nunique()}")

print(f"Windows/file: min={file_sizes.min()}, median={file_sizes.median()}, max={file_sizes.max()}")

print(f"Active classes={active_mask.sum()} of {NUM_CLASSES}")

print(f"Positive windows/class: min={targets[:, active_mask].sum(0).min():.0f}, median={np.median(targets[:, active_mask].sum(0)):.0f}, max={targets[:, active_mask].sum(0).max():.0f}")

CACHE_DIR = Path("/kaggle/working/novel_crg_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)



mel_transform = torchaudio.transforms.MelSpectrogram(

    sample_rate=SAMPLE_RATE,

    n_fft=1024,

    win_length=1024,

    hop_length=640,

    f_min=80,

    f_max=7_800,

    n_mels=64,

    power=2.0,

).to(DEVICE)



def deterministic_fraction(key):

    digest = hashlib.sha256(str(key).encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "little") / 2**64



def read_fixed_clip(path, key, start_seconds=None):

    info = sf.info(path)

    source_frames = round(WINDOW_SECONDS * info.samplerate)

    if start_seconds is None:

        available = max(0, info.frames - source_frames)

        start_frame = round(deterministic_fraction(key) * available)

    else:

        start_frame = round(float(start_seconds) * info.samplerate)

    waveform, source_rate = sf.read(path, start=start_frame, frames=source_frames, dtype="float32", always_2d=True)

    waveform = waveform.mean(axis=1)

    tensor = torch.from_numpy(waveform)

    if source_rate != SAMPLE_RATE:

        tensor = torchaudio.functional.resample(tensor, source_rate, SAMPLE_RATE)

    target_frames = SAMPLE_RATE * WINDOW_SECONDS

    if tensor.numel() < target_frames:

        tensor = F.pad(tensor, (0, target_frames - tensor.numel()))

    return tensor[:target_frames]



@torch.inference_mode()

def waveforms_to_logmels(waveforms):

    batch = torch.stack(waveforms).to(DEVICE, non_blocking=True)

    mels = torch.log(mel_transform(batch).clamp_min(1e-6))

    means = mels.mean(dim=(-2, -1), keepdim=True)

    scales = mels.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)

    return ((mels - means) / scales).unsqueeze(1).half().cpu()



def extract_records(records, cache_path, batch_size=48):

    if cache_path.exists():

        print(f"Loading cache: {cache_path}")

        return torch.load(cache_path, map_location="cpu", weights_only=True)

    chunks = []

    pending = []

    started = time.time()

    for record_index, (path, key, start_seconds) in enumerate(records, start=1):

        pending.append(read_fixed_clip(path, key, start_seconds))

        if len(pending) == batch_size or record_index == len(records):

            chunks.append(waveforms_to_logmels(pending))

            pending.clear()

        if record_index % 500 == 0:

            print(f"  extracted {record_index}/{len(records)} clips in {time.time() - started:.1f}s")

    features = torch.cat(chunks)

    torch.save(features, cache_path)

    print(f"Saved {tuple(features.shape)} to {cache_path} in {time.time() - started:.1f}s")

    return features



train_metadata = pd.read_csv(DATA_ROOT / "train.csv")

train_metadata["primary_label"] = train_metadata["primary_label"].astype(str)

selected_parts = []

for _, class_frame in train_metadata.groupby("primary_label", sort=False):

    selected_parts.append(class_frame.sample(n=min(FOCAL_PER_CLASS, len(class_frame)), random_state=SEED))

focal_df = pd.concat(selected_parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

focal_records = [

    (DATA_ROOT / "train_audio" / row.filename, row.filename, None)

    for row in focal_df.itertuples(index=False)

]

soundscape_records = [

    (DATA_ROOT / "train_soundscapes" / row.filename, f"{row.filename}:{row.start_sec}", row.start_sec)

    for row in windows_df.itertuples(index=False)

]



print(f"Focal pretraining clips={len(focal_records)} across {focal_df['primary_label'].nunique()} classes")

print(f"Soundscape validation clips={len(soundscape_records)}")

source_mels = extract_records(focal_records, CACHE_DIR / f"source_mels_{SAMPLE_RATE}_{FOCAL_PER_CLASS}_fmax7800.pt")

soundscape_mels = extract_records(soundscape_records, CACHE_DIR / f"soundscape_mels_complete_{SAMPLE_RATE}_fmax7800.pt")

source_targets = torch.tensor([class_to_index[label] for label in focal_df["primary_label"]], dtype=torch.long)

print(f"Source tensors: X={tuple(source_mels.shape)}, y={tuple(source_targets.shape)}")

print(f"Soundscape tensors: X={tuple(soundscape_mels.shape)}, y={targets.shape}")

import copy

from torch.utils.data import DataLoader, TensorDataset



class AcousticSourceModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, embedding_dim=128):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False),

            nn.BatchNorm2d(24),

            nn.GELU(),

            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(48),

            nn.GELU(),

            nn.Conv2d(48, 96, 3, stride=2, padding=1, bias=False),

            nn.BatchNorm2d(96),

            nn.GELU(),

            nn.AdaptiveAvgPool2d((1, 1)),

        )

        self.projection = nn.Sequential(

            nn.Flatten(),

            nn.Linear(96, embedding_dim),

            nn.LayerNorm(embedding_dim),

            nn.GELU(),

            nn.Dropout(0.15),

        )

        self.classifier = nn.Linear(embedding_dim, num_classes)



    def forward(self, inputs, return_embedding=False):

        embedding = self.projection(self.encoder(inputs))

        logits = self.classifier(embedding)

        return (logits, embedding) if return_embedding else logits



source_model_path = CACHE_DIR / "acoustic_source_model_focal_earlystop_fmax7800.pt"

source_model = AcousticSourceModel().to(DEVICE)



train_indices = []

validation_indices = []

for _, class_indices in focal_df.groupby("primary_label").groups.items():

    class_indices = np.asarray(sorted(class_indices))

    if len(class_indices) >= 5:

        validation_count = max(1, round(0.2 * len(class_indices)))

        validation_indices.extend(class_indices[-validation_count:])

        train_indices.extend(class_indices[:-validation_count])

    else:

        train_indices.extend(class_indices)

train_indices = torch.tensor(train_indices, dtype=torch.long)

validation_indices = torch.tensor(validation_indices, dtype=torch.long)



source_train_loader = DataLoader(

    TensorDataset(source_mels[train_indices], source_targets[train_indices]),

    batch_size=96,

    shuffle=True,

    generator=torch.Generator().manual_seed(SEED),

    num_workers=2,

    pin_memory=True,

)

source_validation_loader = DataLoader(

    TensorDataset(source_mels[validation_indices], source_targets[validation_indices]),

    batch_size=128,

    shuffle=False,

    num_workers=2,

    pin_memory=True,

)



if source_model_path.exists():

    checkpoint = torch.load(source_model_path, map_location=DEVICE, weights_only=True)

    source_model.load_state_dict(checkpoint["model_state"])

    print(f"Loaded source model selected at focal epoch {checkpoint['best_epoch']}: {source_model_path}")

else:

    optimizer = torch.optim.AdamW(source_model.parameters(), lr=2e-3, weight_decay=2e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_validation_loss = float("inf")

    best_epoch = 0

    best_state = None

    patience = 8

    for epoch in range(1, SOURCE_EPOCHS + 1):

        source_model.train()

        running_loss = 0.0

        correct = 0

        seen = 0

        for batch_mels, batch_targets in source_train_loader:

            batch_mels = batch_mels.to(DEVICE, non_blocking=True).float()

            batch_targets = batch_targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):

                logits = source_model(batch_mels)

                loss = F.cross_entropy(logits, batch_targets, label_smoothing=0.03)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item() * len(batch_targets)

            correct += (logits.argmax(1) == batch_targets).sum().item()

            seen += len(batch_targets)



        source_model.eval()

        validation_loss = 0.0

        validation_correct = 0

        validation_seen = 0

        with torch.inference_mode():

            for batch_mels, batch_targets in source_validation_loader:

                batch_mels = batch_mels.to(DEVICE, non_blocking=True).float()

                batch_targets = batch_targets.to(DEVICE, non_blocking=True)

                logits = source_model(batch_mels)

                loss = F.cross_entropy(logits, batch_targets)

                validation_loss += loss.item() * len(batch_targets)

                validation_correct += (logits.argmax(1) == batch_targets).sum().item()

                validation_seen += len(batch_targets)

        validation_loss /= validation_seen

        if validation_loss < best_validation_loss - 1e-4:

            best_validation_loss = validation_loss

            best_epoch = epoch

            best_state = copy.deepcopy(source_model.state_dict())

        if epoch == 1 or epoch % 5 == 0 or epoch == SOURCE_EPOCHS:

            print(f"epoch={epoch:02d} train_loss={running_loss / seen:.4f} train_acc={correct / seen:.3f} val_loss={validation_loss:.4f} val_acc={validation_correct / validation_seen:.3f}")

        if epoch - best_epoch >= patience:

            print(f"Early stopping at epoch {epoch}; best focal-validation epoch={best_epoch}")

            break

    source_model.load_state_dict(best_state)

    torch.save({"model_state": best_state, "best_epoch": best_epoch, "validation_loss": best_validation_loss}, source_model_path)

    print(f"Saved focal-selected source model: {source_model_path}")



source_model.eval()

embedding_chunks = []

logit_chunks = []

with torch.inference_mode():

    for start in range(0, len(soundscape_mels), 128):

        batch = soundscape_mels[start:start + 128].to(DEVICE).float()

        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):

            batch_logits, batch_embeddings = source_model(batch, return_embedding=True)

        logit_chunks.append(batch_logits.float().cpu())

        embedding_chunks.append(batch_embeddings.float().cpu())

soundscape_source_logits = torch.cat(logit_chunks)

soundscape_embeddings = torch.cat(embedding_chunks)

source_coverage = np.array([label in set(focal_df["primary_label"]) for label in class_names], dtype=np.float32)

print(f"Focal split: train={len(train_indices)}, validation={len(validation_indices)}")

print(f"Frozen source logits={tuple(soundscape_source_logits.shape)}, embeddings={tuple(soundscape_embeddings.shape)}")

print(f"Source-covered targets={source_coverage.sum():.0f}, uncovered targets={(1 - source_coverage).sum():.0f}")

from scipy.special import expit, logit



file_names = sorted(windows_df["filename"].unique())

file_to_rows = {

    filename: windows_df.index[windows_df["filename"] == filename].to_numpy()

    for filename in file_names

}

assert all(len(rows) == 12 for rows in file_to_rows.values())

sequence_rows = np.stack([file_to_rows[filename] for filename in file_names])

sequence_targets = targets[sequence_rows]

file_targets = sequence_targets.max(axis=1)

sequence_embeddings = soundscape_embeddings[sequence_rows]

sequence_source_logits = soundscape_source_logits[sequence_rows]

sequence_sites = torch.tensor(windows_df.loc[sequence_rows[:, 0], "site_index"].to_numpy(), dtype=torch.long)

sequence_hours = torch.tensor(windows_df.loc[sequence_rows[:, 0], "hour"].to_numpy(), dtype=torch.long)



def make_balanced_group_folds(file_level_targets, num_folds=NUM_FOLDS, seed=SEED):

    rng = np.random.default_rng(seed)

    class_totals = file_level_targets.sum(axis=0)

    rarity = (file_level_targets / np.maximum(class_totals, 1)).sum(axis=1)

    jitter = rng.uniform(0, 1e-6, len(file_level_targets))

    order = np.argsort(-(rarity + jitter))

    fold_counts = np.zeros((num_folds, file_level_targets.shape[1]), dtype=np.float64)

    fold_sizes = np.zeros(num_folds, dtype=np.int64)

    assignments = np.full(len(file_level_targets), -1, dtype=np.int64)

    for file_index in order:

        positive = file_level_targets[file_index] > 0

        label_cost = np.array([

            (fold_counts[fold, positive] / np.maximum(class_totals[positive], 1)).sum()

            for fold in range(num_folds)

        ])

        size_cost = fold_sizes / max(1, len(file_level_targets))

        candidate_cost = label_cost + 0.35 * size_cost

        minimum = candidate_cost.min()

        candidates = np.flatnonzero(np.isclose(candidate_cost, minimum))

        chosen = candidates[np.argmin(fold_sizes[candidates])]

        assignments[file_index] = chosen

        fold_counts[chosen] += file_level_targets[file_index]

        fold_sizes[chosen] += 1

    return assignments



fold_ids = make_balanced_group_folds(file_targets)

print("Fold distribution")

for fold in range(NUM_FOLDS):

    selected = fold_ids == fold

    print(f"  fold={fold} files={selected.sum()} windows={selected.sum() * 12} active_file_labels={(file_targets[selected].sum(0) > 0).sum()}")



def grouped_rows(file_indices):

    return sequence_rows[np.asarray(file_indices)].reshape(-1)



def fold_safe_prior(train_file_indices, query_file_indices, alpha=1.0):

    train_rows = grouped_rows(train_file_indices)

    query_rows = grouped_rows(query_file_indices)

    train_y = targets[train_rows]

    global_probability = (train_y.sum(axis=0) + alpha) / (len(train_y) + 2 * alpha)



    def shrunk_probability(mask, parent, smoothing):

        count = int(mask.sum())

        if count == 0:

            return parent

        empirical = (train_y[mask].sum(axis=0) + alpha) / (count + 2 * alpha)

        weight = count / (count + smoothing)

        return weight * empirical + (1 - weight) * parent



    train_meta = windows_df.iloc[train_rows]

    query_meta = windows_df.iloc[query_rows]

    site_priors = {}

    hour_priors = {}

    pair_priors = {}

    for site in query_meta["site"].unique():

        site_priors[site] = shrunk_probability((train_meta["site"].to_numpy() == site), global_probability, 8)

    for hour in query_meta["hour"].unique():

        hour_priors[hour] = shrunk_probability((train_meta["hour"].to_numpy() == hour), global_probability, 8)

    for site, hour in query_meta[["site", "hour"]].drop_duplicates().itertuples(index=False):

        parent = 0.5 * (site_priors[site] + hour_priors[hour])

        mask = (train_meta["site"].to_numpy() == site) & (train_meta["hour"].to_numpy() == hour)

        pair_priors[(site, hour)] = shrunk_probability(mask, parent, 4)

    probabilities = np.stack([pair_priors[(row.site, row.hour)] for row in query_meta.itertuples()])

    return torch.tensor(logit(np.clip(probabilities, 1e-4, 1 - 1e-4)), dtype=torch.float32).reshape(len(query_file_indices), 12, NUM_CLASSES)



def macro_auc(y_true, scores):

    values = []

    for class_index in range(y_true.shape[1]):

        if np.unique(y_true[:, class_index]).size == 2:

            values.append(roc_auc_score(y_true[:, class_index], scores[:, class_index]))

    return float(np.mean(values)), len(values)



def classwise_auc(y_true, scores):

    values = np.full(y_true.shape[1], np.nan, dtype=np.float64)

    for class_index in range(y_true.shape[1]):

        if np.unique(y_true[:, class_index]).size == 2:

            values[class_index] = roc_auc_score(y_true[:, class_index], scores[:, class_index])

    return values



source_auc, source_auc_classes = macro_auc(targets, soundscape_source_logits.numpy())

print(f"Diagnostic source-only pooled AUC (not OOF-trained on soundscapes)={source_auc:.4f} over {source_auc_classes} classes")

class TemporalHybridModel(nn.Module):

    def __init__(self, embedding_dim=128, hidden_dim=96, num_sites=32):

        super().__init__()

        self.embedding_projection = nn.Sequential(

            nn.Linear(embedding_dim, hidden_dim),

            nn.LayerNorm(hidden_dim),

            nn.GELU(),

        )

        self.source_projection = nn.Sequential(

            nn.Linear(NUM_CLASSES, hidden_dim),

            nn.LayerNorm(hidden_dim),

            nn.GELU(),

        )

        self.site_embedding = nn.Embedding(num_sites, 16)

        self.hour_embedding = nn.Embedding(24, 16)

        self.position_embedding = nn.Embedding(12, 16)

        self.input_projection = nn.Linear(hidden_dim * 2 + 48, hidden_dim)

        self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=0.10, batch_first=True)

        self.normalization = nn.LayerNorm(hidden_dim * 2)

        self.residual_head = nn.Sequential(

            nn.Linear(hidden_dim * 2, hidden_dim),

            nn.GELU(),

            nn.Dropout(0.15),

            nn.Linear(hidden_dim, NUM_CLASSES),

        )

        nn.init.zeros_(self.residual_head[-1].weight)

        nn.init.zeros_(self.residual_head[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(-1.0))

        self.prior_scale = nn.Parameter(torch.tensor(0.0))



    def forward(self, embeddings, source_logits, prior_logits, sites, hours):

        batch_size, steps, _ = embeddings.shape

        positions = torch.arange(steps, device=embeddings.device).unsqueeze(0).expand(batch_size, -1)

        metadata = torch.cat([

            self.site_embedding(sites).unsqueeze(1).expand(-1, steps, -1),

            self.hour_embedding(hours).unsqueeze(1).expand(-1, steps, -1),

            self.position_embedding(positions),

        ], dim=-1)

        inputs = torch.cat([

            self.embedding_projection(embeddings),

            self.source_projection(source_logits),

            metadata,

        ], dim=-1)

        temporal, _ = self.temporal(self.input_projection(inputs))

        attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)

        features = self.normalization(temporal + attended)

        correction = self.residual_head(features)

        base = source_logits + torch.sigmoid(self.prior_scale) * prior_logits

        return base + torch.sigmoid(self.residual_scale) * correction



def train_hybrid_fold(train_file_indices, validation_file_indices, fold_number):

    train_file_indices = np.asarray(train_file_indices)

    validation_file_indices = np.asarray(validation_file_indices)

    train_prior = fold_safe_prior(train_file_indices, train_file_indices)

    validation_prior = fold_safe_prior(train_file_indices, validation_file_indices)



    train_x = sequence_embeddings[train_file_indices].float()

    train_source = sequence_source_logits[train_file_indices].float()

    train_y = torch.tensor(sequence_targets[train_file_indices], dtype=torch.float32)

    train_sites = sequence_sites[train_file_indices]

    train_hours = sequence_hours[train_file_indices]



    model = TemporalHybridModel(num_sites=int(windows_df["site_index"].max()) + 1).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)

    positive_counts = train_y.sum(dim=(0, 1))

    negative_counts = train_y.shape[0] * train_y.shape[1] - positive_counts

    positive_weights = (negative_counts / positive_counts.clamp_min(1)).clamp(1.0, 20.0).to(DEVICE)

    generator = torch.Generator().manual_seed(SEED + fold_number)



    for epoch in range(1, HYBRID_EPOCHS + 1):

        model.train()

        order = torch.randperm(len(train_file_indices), generator=generator)

        epoch_loss = 0.0

        for start in range(0, len(order), 12):

            indices = order[start:start + 12]

            embeddings = train_x[indices].to(DEVICE)

            source_logits = train_source[indices].to(DEVICE)

            prior_logits = train_prior[indices].to(DEVICE)

            labels = train_y[indices].to(DEVICE)

            sites = train_sites[indices].to(DEVICE)

            hours = train_hours[indices].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):

                logits = model(embeddings, source_logits, prior_logits, sites, hours)

                classification_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weights)

                distillation_loss = F.mse_loss(logits, source_logits)

                loss = classification_loss + 0.015 * distillation_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            optimizer.step()

            epoch_loss += loss.item() * len(indices)

        if epoch in {1, 10, 20, HYBRID_EPOCHS}:

            print(f"    fold={fold_number} epoch={epoch:02d} loss={epoch_loss / len(order):.4f}")



    model.eval()

    with torch.inference_mode():

        train_hybrid = model(

            train_x.to(DEVICE), train_source.to(DEVICE), train_prior.to(DEVICE),

            train_sites.to(DEVICE), train_hours.to(DEVICE),

        ).float().cpu()

        validation_hybrid = model(

            sequence_embeddings[validation_file_indices].float().to(DEVICE),

            sequence_source_logits[validation_file_indices].float().to(DEVICE),

            validation_prior.to(DEVICE),

            sequence_sites[validation_file_indices].to(DEVICE),

            sequence_hours[validation_file_indices].to(DEVICE),

        ).float().cpu()

    return model, train_hybrid, validation_hybrid, validation_prior



def reliability_gate(train_file_indices, train_hybrid_logits, kappa=8.0):

    train_file_indices = np.asarray(train_file_indices)

    train_labels = sequence_targets[train_file_indices].reshape(-1, NUM_CLASSES)

    hybrid_scores = train_hybrid_logits.numpy().reshape(-1, NUM_CLASSES)

    source_scores = sequence_source_logits[train_file_indices].numpy().reshape(-1, NUM_CLASSES)

    hybrid_auc = classwise_auc(train_labels, hybrid_scores)

    source_auc_values = classwise_auc(train_labels, source_scores)

    reliability_gap = np.nan_to_num(hybrid_auc - source_auc_values, nan=0.0)

    positive_files = file_targets[train_file_indices].sum(axis=0)

    matched = source_coverage

    q = expit(-1.5 + 0.65 * np.log1p(positive_files) - 0.75 * matched + 2.0 * reliability_gap)

    support_weight = positive_files / (positive_files + kappa)

    gate = support_weight * q + (1.0 - support_weight) * (1.0 - matched)

    gate[positive_files == 0] = 1.0 - matched[positive_files == 0]

    return np.clip(gate, 0.0, 1.0), reliability_gap, positive_files



print(f"Temporal hybrid trainable parameters={sum(parameter.numel() for parameter in TemporalHybridModel().parameters() if parameter.requires_grad):,}")

oof_source = np.zeros((len(windows_df), NUM_CLASSES), dtype=np.float32)

oof_prior = np.zeros_like(oof_source)

oof_prior_fusion = np.zeros_like(oof_source)

oof_hybrid = np.zeros_like(oof_source)

oof_gated = np.zeros_like(oof_source)

gate_by_fold = np.zeros((NUM_FOLDS, NUM_CLASSES), dtype=np.float32)

fold_records = []



experiment_started = time.time()

for fold in range(NUM_FOLDS):

    print(f"\n=== Outer fold {fold + 1}/{NUM_FOLDS} ===")

    validation_files = np.flatnonzero(fold_ids == fold)

    train_files = np.flatnonzero(fold_ids != fold)

    torch.manual_seed(SEED + fold)

    torch.cuda.manual_seed_all(SEED + fold)

    model, train_hybrid, validation_hybrid, validation_prior = train_hybrid_fold(train_files, validation_files, fold)

    gate, reliability_gap, positive_files = reliability_gate(train_files, train_hybrid)

    gate_by_fold[fold] = gate



    validation_source = sequence_source_logits[validation_files].numpy()

    validation_prior_np = validation_prior.numpy()

    validation_hybrid_np = validation_hybrid.numpy()

    validation_gated = gate[None, None, :] * validation_hybrid_np + (1.0 - gate[None, None, :]) * validation_source

    validation_prior_fusion = validation_source + 0.5 * validation_prior_np

    rows = grouped_rows(validation_files)

    oof_source[rows] = validation_source.reshape(-1, NUM_CLASSES)

    oof_prior[rows] = validation_prior_np.reshape(-1, NUM_CLASSES)

    oof_prior_fusion[rows] = validation_prior_fusion.reshape(-1, NUM_CLASSES)

    oof_hybrid[rows] = validation_hybrid_np.reshape(-1, NUM_CLASSES)

    oof_gated[rows] = validation_gated.reshape(-1, NUM_CLASSES)



    fold_y = targets[rows]

    metrics = {}

    for name, scores in {

        "source": validation_source,

        "prior": validation_prior_np,

        "prior_fusion": validation_prior_fusion,

        "hybrid": validation_hybrid_np,

        "gated": validation_gated,

    }.items():

        metrics[name], evaluated = macro_auc(fold_y, scores.reshape(-1, NUM_CLASSES))

    fold_records.append({"fold": fold, "files": len(validation_files), "classes": evaluated, **metrics})

    print("    " + "  ".join(f"{name}={value:.4f}" for name, value in metrics.items()))

    print(f"    gate mean={gate.mean():.3f}, covered={gate[source_coverage == 1].mean():.3f}, uncovered={gate[source_coverage == 0].mean():.3f}")

    del model

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()



elapsed = time.time() - experiment_started

fold_results = pd.DataFrame(fold_records)

print(f"\nCompleted five-fold experiment in {elapsed:.1f}s")

display(fold_results.round(4))

def nested_reliability_gate(outer_train_files, outer_fold, inner_folds=3, kappa=4.0):

    outer_train_files = np.asarray(outer_train_files)

    local_targets = file_targets[outer_train_files]

    local_fold_ids = make_balanced_group_folds(local_targets, num_folds=inner_folds, seed=SEED + 100 + outer_fold)

    inner_oof_hybrid = np.zeros((len(outer_train_files), 12, NUM_CLASSES), dtype=np.float32)



    for inner_fold in range(inner_folds):

        inner_validation_local = np.flatnonzero(local_fold_ids == inner_fold)

        inner_train_local = np.flatnonzero(local_fold_ids != inner_fold)

        inner_validation_global = outer_train_files[inner_validation_local]

        inner_train_global = outer_train_files[inner_train_local]

        inner_seed = SEED + 100 + outer_fold * inner_folds + inner_fold

        torch.manual_seed(inner_seed)

        torch.cuda.manual_seed_all(inner_seed)

        _, _, inner_validation_hybrid, _ = train_hybrid_fold(

            inner_train_global,

            inner_validation_global,

            fold_number=100 + outer_fold * inner_folds + inner_fold,

        )

        inner_oof_hybrid[inner_validation_local] = inner_validation_hybrid.numpy()

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()



    inner_labels = sequence_targets[outer_train_files].reshape(-1, NUM_CLASSES)

    inner_source = sequence_source_logits[outer_train_files].numpy().reshape(-1, NUM_CLASSES)

    inner_hybrid = inner_oof_hybrid.reshape(-1, NUM_CLASSES)

    source_auc_values = classwise_auc(inner_labels, inner_source)

    hybrid_auc_values = classwise_auc(inner_labels, inner_hybrid)

    reliability_gap = np.nan_to_num(hybrid_auc_values - source_auc_values, nan=0.0)

    positive_files = file_targets[outer_train_files].sum(axis=0)

    matched = source_coverage



    # Equal inner-fold reliability gives q=0.5; the gap controls which branch is trusted.

    q = expit(0.20 * np.log1p(positive_files) - 0.50 * matched + 8.0 * reliability_gap)

    support_weight = positive_files / (positive_files + kappa)

    gate = support_weight * q + (1.0 - support_weight) * (1.0 - matched)

    gate[positive_files == 0] = 1.0 - matched[positive_files == 0]

    return np.clip(gate, 0.0, 1.0), reliability_gap, inner_oof_hybrid



oof_nested_gated = np.zeros_like(oof_source)

nested_gate_by_fold = np.zeros((NUM_FOLDS, NUM_CLASSES), dtype=np.float32)

nested_records = []

nested_started = time.time()



for fold in range(NUM_FOLDS):

    print(f"\n=== Nested reliability fold {fold + 1}/{NUM_FOLDS} ===")

    validation_files = np.flatnonzero(fold_ids == fold)

    train_files = np.flatnonzero(fold_ids != fold)

    gate, reliability_gap, inner_oof_hybrid = nested_reliability_gate(train_files, fold)

    nested_gate_by_fold[fold] = gate

    rows = grouped_rows(validation_files)

    validation_source = sequence_source_logits[validation_files].numpy()

    validation_hybrid = oof_hybrid[rows].reshape(len(validation_files), 12, NUM_CLASSES)

    validation_gated = gate[None, None, :] * validation_hybrid + (1.0 - gate[None, None, :]) * validation_source

    oof_nested_gated[rows] = validation_gated.reshape(-1, NUM_CLASSES)

    fold_auc, classes = macro_auc(targets[rows], validation_gated.reshape(-1, NUM_CLASSES))

    nested_records.append({

        "fold": fold,

        "files": len(validation_files),

        "classes": classes,

        "nested_gated": fold_auc,

        "gate_mean": gate.mean(),

        "gate_covered": gate[source_coverage == 1].mean(),

        "gate_uncovered": gate[source_coverage == 0].mean(),

    })

    print(f"    nested_gated={fold_auc:.4f} gate_mean={gate.mean():.3f} covered={gate[source_coverage == 1].mean():.3f} uncovered={gate[source_coverage == 0].mean():.3f}")



nested_elapsed = time.time() - nested_started

nested_results = pd.DataFrame(nested_records)

print(f"\nCompleted nested reliability evaluation in {nested_elapsed:.1f}s")

display(nested_results.round(4))

score_sets = {

    "Focal source CNN": oof_source,

    "Fold-safe site-hour prior": oof_prior,

    "Source + prior": oof_prior_fusion,

    "Temporal hybrid": oof_hybrid,

    "Fixed reliability gate (pilot)": oof_gated,

    "Nested reliability gate": oof_nested_gated,

}



summary_rows = []

for name, scores in score_sets.items():

    auc, evaluated_classes = macro_auc(targets, scores)

    class_auc = classwise_auc(targets, scores)

    covered_active = active_mask & (source_coverage == 1)

    uncovered_active = active_mask & (source_coverage == 0)

    summary_rows.append({

        "model": name,

        "macro_oof_auc": auc,

        "evaluated_classes": evaluated_classes,

        "covered_class_auc": float(np.nanmean(class_auc[covered_active])) if covered_active.any() else np.nan,

        "uncovered_class_auc": float(np.nanmean(class_auc[uncovered_active])) if uncovered_active.any() else np.nan,

    })



summary_df = pd.DataFrame(summary_rows)

print("Pooled grouped out-of-fold results")

display(summary_df.round(4))

print(f"Active source-covered classes={int((active_mask & (source_coverage == 1)).sum())}")

print(f"Active source-uncovered classes={int((active_mask & (source_coverage == 0)).sum())}")



fold_comparison = fold_results[["fold", "source", "prior", "prior_fusion", "hybrid", "gated"]].merge(

    nested_results[["fold", "nested_gated"]], on="fold"

)

print("\nMean and standard deviation across folds (folds contain different valid-class sets)")

fold_statistics = pd.DataFrame({

    "mean": fold_comparison.drop(columns="fold").mean(),

    "std": fold_comparison.drop(columns="fold").std(ddof=1),

})

display(fold_statistics.round(4))



def cluster_bootstrap_auc(scores, repetitions=500, seed=SEED):

    rng = np.random.default_rng(seed)

    bootstrap_values = []

    file_count = len(file_names)

    for _ in range(repetitions):

        sampled_files = rng.integers(0, file_count, size=file_count)

        sampled_rows = grouped_rows(sampled_files)

        value, _ = macro_auc(targets[sampled_rows], scores[sampled_rows])

        bootstrap_values.append(value)

    return np.quantile(bootstrap_values, [0.025, 0.975])



print("\nRecording-cluster bootstrap 95% intervals")

interval_rows = []

for name in ["Focal source CNN", "Fold-safe site-hour prior", "Source + prior", "Temporal hybrid", "Nested reliability gate"]:

    auc = summary_df.loc[summary_df["model"] == name, "macro_oof_auc"].iloc[0]

    lower, upper = cluster_bootstrap_auc(score_sets[name])

    interval_rows.append({"model": name, "macro_oof_auc": auc, "ci_2.5%": lower, "ci_97.5%": upper})

interval_df = pd.DataFrame(interval_rows)

display(interval_df.round(4))



results_path = Path("/kaggle/working/novel_crg_results.csv")

interval_df.to_csv(results_path, index=False)

print(f"Saved measured result summary: {results_path}")

def paired_cluster_bootstrap_delta(candidate, reference, repetitions=250, seed=SEED + 1):

    rng = np.random.default_rng(seed)

    differences = []

    for _ in range(repetitions):

        sampled_files = rng.integers(0, len(file_names), size=len(file_names))

        sampled_rows = grouped_rows(sampled_files)

        candidate_auc, _ = macro_auc(targets[sampled_rows], candidate[sampled_rows])

        reference_auc, _ = macro_auc(targets[sampled_rows], reference[sampled_rows])

        differences.append(candidate_auc - reference_auc)

    return np.quantile(differences, [0.025, 0.5, 0.975])



delta_rows = []

for reference_name in ["Focal source CNN", "Source + prior", "Temporal hybrid", "Fold-safe site-hour prior"]:

    lower, median, upper = paired_cluster_bootstrap_delta(oof_nested_gated, score_sets[reference_name])

    observed = macro_auc(targets, oof_nested_gated)[0] - macro_auc(targets, score_sets[reference_name])[0]

    delta_rows.append({

        "comparison": f"Nested gate - {reference_name}",

        "observed_delta": observed,

        "bootstrap_median": median,

        "ci_2.5%": lower,

        "ci_97.5%": upper,

    })

delta_df = pd.DataFrame(delta_rows)

print("Paired recording-cluster bootstrap differences")

display(delta_df.round(4))



np.savez_compressed(

    "/kaggle/working/novel_crg_oof_predictions.npz",

    targets=targets,

    fold_ids=fold_ids,

    source=oof_source,

    prior=oof_prior,

    source_prior=oof_prior_fusion,

    temporal_hybrid=oof_hybrid,

    nested_gate=oof_nested_gated,

    nested_gate_weights=nested_gate_by_fold,

)

run_manifest = {

    "seed": SEED,

    "sample_rate": SAMPLE_RATE,

    "window_seconds": WINDOW_SECONDS,

    "focal_examples": len(focal_df),

    "focal_classes": int(focal_df["primary_label"].nunique()),

    "complete_soundscape_files": len(file_names),

    "soundscape_windows": len(windows_df),

    "active_classes": int(active_mask.sum()),

    "outer_folds": NUM_FOLDS,

    "inner_folds": 3,

    "source_checkpoint": str(source_model_path),

    "hybrid_parameters": sum(parameter.numel() for parameter in TemporalHybridModel().parameters() if parameter.requires_grad),

}

with open("/kaggle/working/novel_crg_manifest.json", "w", encoding="utf-8") as handle:

    json.dump(run_manifest, handle, indent=2)

print("Saved OOF predictions and run manifest under /kaggle/working")

from scipy.special import softmax

from scipy.stats import rankdata



def rank_normalize(scores):

    scores = np.asarray(scores)

    normalized = np.empty_like(scores, dtype=np.float32)

    for class_index in range(scores.shape[1]):

        normalized[:, class_index] = rankdata(scores[:, class_index], method="average") / (len(scores) + 1.0)

    return normalized



def nested_tri_branch_weights(outer_train_files, outer_fold, inner_folds=3, kappa=4.0, temperature=5.0):

    outer_train_files = np.asarray(outer_train_files)

    local_fold_ids = make_balanced_group_folds(

        file_targets[outer_train_files], num_folds=inner_folds, seed=SEED + 500 + outer_fold

    )

    inner_source = np.zeros((len(outer_train_files), 12, NUM_CLASSES), dtype=np.float32)

    inner_prior = np.zeros_like(inner_source)

    inner_hybrid = np.zeros_like(inner_source)



    for inner_fold in range(inner_folds):

        validation_local = np.flatnonzero(local_fold_ids == inner_fold)

        train_local = np.flatnonzero(local_fold_ids != inner_fold)

        validation_global = outer_train_files[validation_local]

        train_global = outer_train_files[train_local]

        inner_seed = SEED + 500 + outer_fold * inner_folds + inner_fold

        torch.manual_seed(inner_seed)

        torch.cuda.manual_seed_all(inner_seed)

        _, _, validation_hybrid, validation_prior = train_hybrid_fold(

            train_global, validation_global, fold_number=500 + outer_fold * inner_folds + inner_fold

        )

        inner_source[validation_local] = sequence_source_logits[validation_global].numpy()

        inner_prior[validation_local] = validation_prior.numpy()

        inner_hybrid[validation_local] = validation_hybrid.numpy()



    inner_labels = sequence_targets[outer_train_files].reshape(-1, NUM_CLASSES)

    branch_scores = [inner_source, inner_prior, inner_hybrid]

    branch_aucs = np.stack([

        classwise_auc(inner_labels, scores.reshape(-1, NUM_CLASSES)) for scores in branch_scores

    ])

    branch_aucs = np.nan_to_num(branch_aucs, nan=0.5)

    positive_files = file_targets[outer_train_files].sum(axis=0)

    support = positive_files / (positive_files + kappa)

    shrunk_reliability = 0.5 + support[None, :] * (branch_aucs - 0.5)

    shrunk_reliability[0, source_coverage == 0] = -1e6

    weights = softmax(temperature * shrunk_reliability, axis=0)

    return weights.astype(np.float32), branch_aucs



oof_tri_gate = np.zeros_like(oof_source)

tri_weights_by_fold = np.zeros((NUM_FOLDS, 3, NUM_CLASSES), dtype=np.float32)

tri_records = []

tri_started = time.time()



for fold in range(NUM_FOLDS):

    print(f"\n=== Tri-branch reliability fold {fold + 1}/{NUM_FOLDS} ===")

    validation_files = np.flatnonzero(fold_ids == fold)

    train_files = np.flatnonzero(fold_ids != fold)

    weights, inner_branch_aucs = nested_tri_branch_weights(train_files, fold)

    tri_weights_by_fold[fold] = weights

    rows = grouped_rows(validation_files)

    branches = [oof_source[rows], oof_prior[rows], oof_hybrid[rows]]

    ranked_branches = np.stack([rank_normalize(branch) for branch in branches])

    validation_scores = (weights[:, None, :] * ranked_branches).sum(axis=0)

    oof_tri_gate[rows] = validation_scores

    fold_auc, classes = macro_auc(targets[rows], validation_scores)

    tri_records.append({

        "fold": fold,

        "files": len(validation_files),

        "classes": classes,

        "tri_gate": fold_auc,

        "source_weight": weights[0].mean(),

        "prior_weight": weights[1].mean(),

        "hybrid_weight": weights[2].mean(),

    })

    print(f"    tri_gate={fold_auc:.4f} mean weights: source={weights[0].mean():.3f}, prior={weights[1].mean():.3f}, hybrid={weights[2].mean():.3f}")



tri_results = pd.DataFrame(tri_records)

tri_auc, tri_classes = macro_auc(targets, oof_tri_gate)

print(f"\nTri-branch pooled OOF AUC={tri_auc:.4f} over {tri_classes} classes; elapsed={time.time() - tri_started:.1f}s")

display(tri_results.round(4))

tri_delta_rows = []

for reference_name in ["Fold-safe site-hour prior", "Temporal hybrid", "Nested reliability gate"]:

    lower, median, upper = paired_cluster_bootstrap_delta(

        oof_tri_gate, score_sets[reference_name], repetitions=250, seed=SEED + 20

    )

    observed = macro_auc(targets, oof_tri_gate)[0] - macro_auc(targets, score_sets[reference_name])[0]

    tri_delta_rows.append({

        "comparison": f"Tri-branch gate - {reference_name}",

        "observed_delta": observed,

        "bootstrap_median": median,

        "ci_2.5%": lower,

        "ci_97.5%": upper,

    })

tri_delta_df = pd.DataFrame(tri_delta_rows)

print("Paired recording-cluster bootstrap for tri-branch gate")

display(tri_delta_df.round(4))



np.savez_compressed(

    "/kaggle/working/novel_crg_oof_predictions.npz",

    targets=targets,

    fold_ids=fold_ids,

    source=oof_source,

    prior=oof_prior,

    source_prior=oof_prior_fusion,

    temporal_hybrid=oof_hybrid,

    nested_binary_gate=oof_nested_gated,

    tri_branch_gate=oof_tri_gate,

    binary_gate_weights=nested_gate_by_fold,

    tri_branch_weights=tri_weights_by_fold,

)

tri_summary = pd.DataFrame([

    {"model": name, "macro_oof_auc": macro_auc(targets, scores)[0]}

    for name, scores in {

        "Focal source CNN": oof_source,

        "Fold-safe site-hour prior": oof_prior,

        "Source + prior": oof_prior_fusion,

        "Temporal hybrid": oof_hybrid,

        "Nested binary gate": oof_nested_gated,

        "Tri-branch reliability gate": oof_tri_gate,

    }.items()

])

tri_summary.to_csv("/kaggle/working/novel_crg_results.csv", index=False)

tri_delta_df.to_csv("/kaggle/working/novel_crg_paired_deltas.csv", index=False)

print("Updated final CSV and OOF artifacts under /kaggle/working")

def foldwise_rank_scores(scores):

    ranked = np.zeros_like(scores, dtype=np.float32)

    for fold in range(NUM_FOLDS):

        validation_files = np.flatnonzero(fold_ids == fold)

        rows = grouped_rows(validation_files)

        ranked[rows] = rank_normalize(scores[rows])

    return ranked



ranked_source_oof = foldwise_rank_scores(oof_source)

ranked_prior_oof = foldwise_rank_scores(oof_prior)

ranked_hybrid_oof = foldwise_rank_scores(oof_hybrid)

uniform_tri_oof = (ranked_source_oof + ranked_prior_oof + ranked_hybrid_oof) / 3.0



fairness_rows = []

for name, scores in {

    "Ranked source": ranked_source_oof,

    "Ranked prior": ranked_prior_oof,

    "Ranked temporal hybrid": ranked_hybrid_oof,

    "Uniform tri-branch rank average": uniform_tri_oof,

    "Reliability-weighted tri-branch": oof_tri_gate,

}.items():

    score, classes = macro_auc(targets, scores)

    fairness_rows.append({"method": name, "macro_oof_auc": score, "classes": classes})

fairness_df = pd.DataFrame(fairness_rows)

print("Rank-normalized fairness comparison")

display(fairness_df.round(4))



fair_delta_rows = []

for reference_name, reference_scores in {

    "Ranked prior": ranked_prior_oof,

    "Ranked temporal hybrid": ranked_hybrid_oof,

    "Uniform tri-branch rank average": uniform_tri_oof,

}.items():

    lower, median, upper = paired_cluster_bootstrap_delta(

        oof_tri_gate, reference_scores, repetitions=250, seed=SEED + 30

    )

    observed = macro_auc(targets, oof_tri_gate)[0] - macro_auc(targets, reference_scores)[0]

    fair_delta_rows.append({

        "comparison": f"Reliability tri-gate - {reference_name}",

        "observed_delta": observed,

        "bootstrap_median": median,

        "ci_2.5%": lower,

        "ci_97.5%": upper,

    })

fair_delta_df = pd.DataFrame(fair_delta_rows)

display(fair_delta_df.round(4))

fairness_df.to_csv("/kaggle/working/novel_crg_ranked_results.csv", index=False)

fair_delta_df.to_csv("/kaggle/working/novel_crg_ranked_paired_deltas.csv", index=False)

np.savez_compressed(

    "/kaggle/working/novel_crg_final_oof_predictions.npz",

    targets=targets,

    fold_ids=fold_ids,

    ranked_source=ranked_source_oof,

    ranked_prior=ranked_prior_oof,

    ranked_temporal=ranked_hybrid_oof,

    uniform_tri_branch=uniform_tri_oof,

    reliability_tri_branch=oof_tri_gate,

    tri_branch_weights=tri_weights_by_fold,

)

print("Saved fair ranked baselines, paired deltas, and final OOF arrays under /kaggle/working")

import timm



BASELINE_MODELS = {

    "MobileNetV3-Small": "mobilenetv3_small_050",

    "EfficientNet-B0": "efficientnet_b0",

    "ResNet-18": "resnet18",

}

BASELINE_SOURCE_EPOCHS = 7

BASELINE_FINETUNE_EPOCHS = 20



def create_acoustic_baseline(model_name):

    return timm.create_model(

        model_name,

        pretrained=False,

        in_chans=1,

        num_classes=NUM_CLASSES,

        global_pool="avg",

    )



def train_focal_initialization(display_name, model_name):

    checkpoint_path = CACHE_DIR / f"{model_name}_focal_{BASELINE_SOURCE_EPOCHS}ep.pt"

    torch.manual_seed(SEED)

    torch.cuda.manual_seed_all(SEED)

    model = create_acoustic_baseline(model_name).to(DEVICE)

    if checkpoint_path.exists():

        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

        print(f"{display_name}: loaded focal checkpoint")

        return model.cpu().state_dict()



    loader = DataLoader(

        TensorDataset(source_mels, source_targets),

        batch_size=64,

        shuffle=True,

        generator=torch.Generator().manual_seed(SEED),

        num_workers=2,

        pin_memory=True,

    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)

    for epoch in range(1, BASELINE_SOURCE_EPOCHS + 1):

        model.train()

        total_loss = 0.0

        correct = 0

        seen = 0

        for batch_mels, batch_targets in loader:

            batch_mels = batch_mels.to(DEVICE, non_blocking=True).float()

            batch_targets = batch_targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):

                logits = model(batch_mels)

                loss = F.cross_entropy(logits, batch_targets, label_smoothing=0.03)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            optimizer.step()

            total_loss += loss.item() * len(batch_targets)

            correct += (logits.argmax(1) == batch_targets).sum().item()

            seen += len(batch_targets)

        if epoch in {1, BASELINE_SOURCE_EPOCHS}:

            print(f"{display_name}: focal epoch={epoch} loss={total_loss / seen:.4f} accuracy={correct / seen:.3f}")

    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    torch.save(cpu_state, checkpoint_path)

    del model

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

    return cpu_state



def train_baseline_outer_folds(display_name, model_name, initial_state):

    predictions = np.zeros((len(windows_df), NUM_CLASSES), dtype=np.float32)

    fold_metrics = []

    parameter_count = None

    for fold in range(NUM_FOLDS):

        train_files = np.flatnonzero(fold_ids != fold)

        validation_files = np.flatnonzero(fold_ids == fold)

        train_rows = grouped_rows(train_files)

        validation_rows = grouped_rows(validation_files)

        torch.manual_seed(SEED + 1000 + fold)

        torch.cuda.manual_seed_all(SEED + 1000 + fold)

        model = create_acoustic_baseline(model_name).to(DEVICE)

        model.load_state_dict(initial_state)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())



        train_y = torch.tensor(targets[train_rows], dtype=torch.float32)

        positive_counts = train_y.sum(0)

        negative_counts = len(train_y) - positive_counts

        positive_weights = (negative_counts / positive_counts.clamp_min(1)).clamp(1.0, 20.0).to(DEVICE)

        loader = DataLoader(

            TensorDataset(soundscape_mels[train_rows], train_y),

            batch_size=64,

            shuffle=True,

            generator=torch.Generator().manual_seed(SEED + 1000 + fold),

            num_workers=2,

            pin_memory=True,

        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-3)

        for epoch in range(1, BASELINE_FINETUNE_EPOCHS + 1):

            model.train()

            total_loss = 0.0

            for batch_mels, batch_targets in loader:

                batch_mels = batch_mels.to(DEVICE, non_blocking=True).float()

                batch_targets = batch_targets.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):

                    logits = model(batch_mels)

                    loss = F.binary_cross_entropy_with_logits(logits, batch_targets, pos_weight=positive_weights)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                optimizer.step()

                total_loss += loss.item() * len(batch_targets)



        model.eval()

        fold_predictions = []

        with torch.inference_mode():

            for start in range(0, len(validation_rows), 128):

                batch_rows = validation_rows[start:start + 128]

                batch_mels = soundscape_mels[batch_rows].to(DEVICE).float()

                fold_predictions.append(model(batch_mels).float().cpu().numpy())

        fold_predictions = np.concatenate(fold_predictions)

        predictions[validation_rows] = fold_predictions

        fold_auc, classes = macro_auc(targets[validation_rows], fold_predictions)

        fold_metrics.append(fold_auc)

        print(f"{display_name}: fold={fold + 1} loss={total_loss / len(train_rows):.4f} AUC={fold_auc:.4f} classes={classes}")

        del model

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()



    ranked_predictions = foldwise_rank_scores(predictions)

    pooled_auc, classes = macro_auc(targets, ranked_predictions)

    return predictions, ranked_predictions, pooled_auc, classes, parameter_count, fold_metrics



print("Baseline parameter counts")

for display_name, model_name in BASELINE_MODELS.items():

    probe = create_acoustic_baseline(model_name)

    print(f"  {display_name}: {sum(parameter.numel() for parameter in probe.parameters()):,}")

    del probe

architecture_oof = {}

architecture_ranked_oof = {}

architecture_records = []

architecture_started = time.time()



for display_name, model_name in BASELINE_MODELS.items():

    print(f"\n=== {display_name} ===")

    initial_state = train_focal_initialization(display_name, model_name)

    raw_predictions, ranked_predictions, pooled_auc, classes, parameters, fold_aucs = train_baseline_outer_folds(

        display_name, model_name, initial_state

    )

    architecture_oof[display_name] = raw_predictions

    architecture_ranked_oof[display_name] = ranked_predictions

    architecture_records.append({

        "model": display_name,

        "parameters": parameters,

        "macro_oof_auc": pooled_auc,

        "evaluated_classes": classes,

        "fold_mean_auc": float(np.mean(fold_aucs)),

        "fold_std_auc": float(np.std(fold_aucs, ddof=1)),

    })

    np.savez_compressed(

        CACHE_DIR / f"{model_name}_grouped_oof.npz",

        raw_predictions=raw_predictions,

        ranked_predictions=ranked_predictions,

        targets=targets,

        fold_ids=fold_ids,

    )

    print(f"{display_name}: pooled ranked OOF AUC={pooled_auc:.4f}")



architecture_results = pd.DataFrame(architecture_records)

print(f"\nCompleted architecture baselines in {time.time() - architecture_started:.1f}s")

display(architecture_results.round(4))

architecture_results.to_csv("/kaggle/working/novel_architecture_baselines.csv", index=False)

architecture_delta_rows = []

for display_name in BASELINE_MODELS:

    baseline_scores = architecture_ranked_oof[display_name]

    lower, median, upper = paired_cluster_bootstrap_delta(

        oof_tri_gate, baseline_scores, repetitions=250, seed=SEED + 40

    )

    observed = macro_auc(targets, oof_tri_gate)[0] - macro_auc(targets, baseline_scores)[0]

    architecture_delta_rows.append({

        "comparison": f"Proposed - {display_name}",

        "observed_delta": observed,

        "bootstrap_median": median,

        "ci_2.5%": lower,

        "ci_97.5%": upper,

    })



architecture_delta_df = pd.DataFrame(architecture_delta_rows)

source_parameter_count = sum(parameter.numel() for parameter in source_model.parameters())

temporal_parameter_count = sum(parameter.numel() for parameter in TemporalHybridModel().parameters())

proposed_parameter_count = source_parameter_count + temporal_parameter_count

print("Paired recording-cluster bootstrap against standard acoustic baselines")

display(architecture_delta_df.round(4))

print(f"Source CNN parameters={source_parameter_count:,}")

print(f"Temporal head parameters={temporal_parameter_count:,}")

print(f"Proposed neural parameters={proposed_parameter_count:,} (prior and gate are non-neural)")



architecture_delta_df.to_csv("/kaggle/working/novel_architecture_paired_deltas.csv", index=False)

np.savez_compressed(

    "/kaggle/working/novel_architecture_oof_predictions.npz",

    targets=targets,

    fold_ids=fold_ids,

    mobilenetv3_small=architecture_ranked_oof["MobileNetV3-Small"],

    efficientnet_b0=architecture_ranked_oof["EfficientNet-B0"],

    resnet18=architecture_ranked_oof["ResNet-18"],

    proposed=oof_tri_gate,

)

from sklearn.linear_model import LogisticRegression


def shrink_to_chance(auc_values, positive_files, kappa):
    support = positive_files / (positive_files + kappa)
    return 0.5 + support * (np.nan_to_num(auc_values, nan=0.5) - 0.5)


def compute_inner_branch_arrays(outer_train_files, outer_fold, inner_folds=3, seed_offset=900):
    outer_train_files = np.asarray(outer_train_files)
    local_fold_ids = make_balanced_group_folds(
        file_targets[outer_train_files], num_folds=inner_folds, seed=SEED + seed_offset + outer_fold
    )
    inner_source = np.zeros((len(outer_train_files), 12, NUM_CLASSES), dtype=np.float32)
    inner_prior = np.zeros_like(inner_source)
    inner_hybrid = np.zeros_like(inner_source)

    for inner_fold in range(inner_folds):
        validation_local = np.flatnonzero(local_fold_ids == inner_fold)
        train_local = np.flatnonzero(local_fold_ids != inner_fold)
        validation_global = outer_train_files[validation_local]
        train_global = outer_train_files[train_local]
        inner_seed = SEED + seed_offset + outer_fold * inner_folds + inner_fold
        torch.manual_seed(inner_seed)
        torch.cuda.manual_seed_all(inner_seed)
        _, _, validation_hybrid, validation_prior = train_hybrid_fold(
            train_global, validation_global, fold_number=seed_offset + outer_fold * inner_folds + inner_fold
        )
        inner_source[validation_local] = sequence_source_logits[validation_global].numpy()
        inner_prior[validation_local] = validation_prior.numpy()
        inner_hybrid[validation_local] = validation_hybrid.numpy()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "source": inner_source,
        "prior": inner_prior,
        "hybrid": inner_hybrid,
        "labels": sequence_targets[outer_train_files],
        "positive_files": file_targets[outer_train_files].sum(axis=0),
    }


def tri_branch_weights_from_inner(inner, kappa=4.0, gamma=5.0, mask_mode="hard"):
    order = ["source", "prior", "hybrid"]
    labels_flat = inner["labels"].reshape(-1, NUM_CLASSES)
    raw_auc = {b: classwise_auc(labels_flat, inner[b].reshape(-1, NUM_CLASSES)) for b in order}
    shrunk = np.stack([shrink_to_chance(raw_auc[b], inner["positive_files"], kappa) for b in order])
    if mask_mode == "hard":
        shrunk[0, source_coverage == 0] = -1e6
    elif mask_mode == "soft":
        floor = np.nanpercentile(shrunk[0], 10)
        shrunk[0, source_coverage == 0] = np.minimum(shrunk[0, source_coverage == 0], floor)
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    weights = softmax(gamma * shrunk, axis=0)
    return weights.astype(np.float32), raw_auc


def global_weighted_fusion_weights(raw_auc, positive_files, kappa=4.0, gamma=5.0):
    order = ["source", "prior", "hybrid"]
    pooled_auc = np.array([np.nanmean(raw_auc[b]) for b in order])
    shrink = positive_files.mean() / (positive_files.mean() + kappa)
    shrunk = 0.5 + shrink * (pooled_auc - 0.5)
    weights = softmax(gamma * shrunk)
    return {branch: float(weight) for branch, weight in zip(order, weights)}


def best_branch_per_class(raw_auc, positive_files, min_support=4, kappa=4.0):
    order = ["source", "prior", "hybrid"]
    shrunk = np.stack([shrink_to_chance(raw_auc[b], positive_files, kappa) for b in order])
    pick = np.argmax(shrunk, axis=0)
    pick[positive_files < min_support] = order.index("prior")
    return pick


def fit_logistic_stack(inner, min_count=3):
    order = ["source", "prior", "hybrid"]
    inner_ranked = {b: rank_normalize(inner[b].reshape(-1, NUM_CLASSES)) for b in order}
    inner_labels_flat = inner["labels"].reshape(-1, NUM_CLASSES)
    models = {}
    for class_index in range(NUM_CLASSES):
        y = inner_labels_flat[:, class_index]
        if len(np.unique(y)) < 2 or y.sum() < min_count or (len(y) - y.sum()) < min_count:
            continue
        features = np.column_stack([inner_ranked[b][:, class_index] for b in order])
        model = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced")
        model.fit(features, y)
        models[class_index] = model
    return models


def apply_logistic_stack(models, rows, branch_arrays, fallback):
    order = ["source", "prior", "hybrid"]
    out = fallback[rows].copy()
    for class_index, model in models.items():
        features = np.column_stack([branch_arrays[b][rows, class_index] for b in order])
        out[:, class_index] = model.predict_proba(features)[:, 1]
    return out


def ensemble_selection_weights(inner, max_rounds=30, tol=1e-4):
    order = ["source", "prior", "hybrid"]
    inner_ranked = [rank_normalize(inner[b].reshape(-1, NUM_CLASSES)) for b in order]
    labels_flat = inner["labels"].reshape(-1, NUM_CLASSES)
    counts = np.zeros(3, dtype=np.int64)
    current_sum = np.zeros_like(inner_ranked[0])
    best_auc = -np.inf
    history = []
    for _ in range(max_rounds):
        round_best = None
        for candidate_index in range(3):
            trial_sum = current_sum + inner_ranked[candidate_index]
            trial_avg = trial_sum / (counts.sum() + 1)
            auc, _ = macro_auc(labels_flat, trial_avg)
            if round_best is None or auc > round_best[1]:
                round_best = (candidate_index, auc, trial_sum)
        if round_best[1] <= best_auc + tol:
            break
        counts[round_best[0]] += 1
        current_sum = round_best[2]
        best_auc = round_best[1]
        history.append(best_auc)
    if counts.sum() == 0:
        counts[order.index("prior")] = 1
    weights = counts / counts.sum()
    return {branch: float(weight) for branch, weight in zip(order, weights)}, history


print("Reviewer-response helper functions defined")

KAPPA_GRID = [2.0, 4.0, 8.0]
GAMMA_GRID = [2.0, 5.0, 8.0]
MIN_SUPPORT_BEST_BRANCH = 4
LOGISTIC_MIN_COUNT = 3
ENSEMBLE_MAX_ROUNDS = 30

branch_arrays = {"source": ranked_source_oof, "prior": ranked_prior_oof, "hybrid": ranked_hybrid_oof}

oof_global_weight = np.zeros_like(oof_source)
oof_best_branch = np.zeros_like(oof_source)
oof_logistic_stack = np.zeros_like(oof_source)
oof_ensemble_selection = np.zeros_like(oof_source)
oof_soft_mask_gate = np.zeros_like(oof_source)

undefined_auc_counts = []
global_weight_records = []
ensemble_weight_records = []
best_branch_picks = np.full((NUM_FOLDS, NUM_CLASSES), -1, dtype=np.int64)
grid_auc = {(kappa, gamma): [] for kappa in KAPPA_GRID for gamma in GAMMA_GRID}

response_started = time.time()
for fold in range(NUM_FOLDS):
    print(f"\n=== Reviewer-response inner pass, outer fold {fold + 1}/{NUM_FOLDS} ===")
    validation_files = np.flatnonzero(fold_ids == fold)
    train_files = np.flatnonzero(fold_ids != fold)
    rows = grouped_rows(validation_files)

    inner = compute_inner_branch_arrays(train_files, fold)
    labels_flat = inner["labels"].reshape(-1, NUM_CLASSES)
    raw_auc = {b: classwise_auc(labels_flat, inner[b].reshape(-1, NUM_CLASSES)) for b in ["source", "prior", "hybrid"]}
    undefined_auc_counts.append({
        "fold": fold,
        "source": int(np.isnan(raw_auc["source"]).sum()),
        "prior": int(np.isnan(raw_auc["prior"]).sum()),
        "hybrid": int(np.isnan(raw_auc["hybrid"]).sum()),
    })

    global_weights = global_weighted_fusion_weights(raw_auc, inner["positive_files"])
    global_weight_records.append({"fold": fold, **global_weights})
    oof_global_weight[rows] = sum(global_weights[b] * branch_arrays[b][rows] for b in ["source", "prior", "hybrid"])

    pick = best_branch_per_class(raw_auc, inner["positive_files"], min_support=MIN_SUPPORT_BEST_BRANCH)
    best_branch_picks[fold] = pick
    for branch_index, branch_name in enumerate(["source", "prior", "hybrid"]):
        class_mask = pick == branch_index
        if class_mask.any():
            oof_best_branch[np.ix_(rows, class_mask)] = branch_arrays[branch_name][np.ix_(rows, class_mask)]

    logistic_models = fit_logistic_stack(inner, min_count=LOGISTIC_MIN_COUNT)
    oof_logistic_stack[rows] = apply_logistic_stack(logistic_models, rows, branch_arrays, fallback=ranked_prior_oof)

    ensemble_weights, ensemble_history = ensemble_selection_weights(inner, max_rounds=ENSEMBLE_MAX_ROUNDS)
    ensemble_weight_records.append({"fold": fold, "rounds": len(ensemble_history), **ensemble_weights})
    oof_ensemble_selection[rows] = sum(ensemble_weights[b] * branch_arrays[b][rows] for b in ["source", "prior", "hybrid"])

    ranked_stack = np.stack([branch_arrays[b][rows] for b in ["source", "prior", "hybrid"]])
    soft_weights, _ = tri_branch_weights_from_inner(inner, kappa=4.0, gamma=5.0, mask_mode="soft")
    oof_soft_mask_gate[rows] = (soft_weights[:, None, :] * ranked_stack).sum(axis=0)

    for kappa in KAPPA_GRID:
        for gamma in GAMMA_GRID:
            grid_weights, _ = tri_branch_weights_from_inner(inner, kappa=kappa, gamma=gamma, mask_mode="hard")
            grid_scores = (grid_weights[:, None, :] * ranked_stack).sum(axis=0)
            fold_auc, _ = macro_auc(targets[rows], grid_scores)
            grid_auc[(kappa, gamma)].append(fold_auc)

    print(f"    undefined inner AUC source/prior/hybrid = {undefined_auc_counts[-1]['source']}/{undefined_auc_counts[-1]['prior']}/{undefined_auc_counts[-1]['hybrid']}")
    print(f"    global weights={global_weights}")
    print(f"    ensemble weights={ensemble_weights} ({len(ensemble_history)} greedy rounds)")
    print(f"    logistic models fitted for {len(logistic_models)}/{NUM_CLASSES} classes")

print(f"\nCompleted reviewer-response inner pass in {time.time() - response_started:.1f}s")

new_variant_scores = {
    "Prior + temporal average": (ranked_prior_oof + ranked_hybrid_oof) / 2.0,
    "Global (non-classwise) AUC-weighted fusion": oof_global_weight,
    "Best-branch-per-class selection": oof_best_branch,
    "Per-class logistic stacking": oof_logistic_stack,
    "Greedy ensemble selection": oof_ensemble_selection,
    "Nested tri-branch gate, soft mask": oof_soft_mask_gate,
}

reviewer_response_rows = []
for name, scores in new_variant_scores.items():
    auc, classes = macro_auc(targets, scores)
    reviewer_response_rows.append({"method": name, "macro_oof_auc": auc, "classes": classes})
reviewer_response_df = pd.DataFrame(reviewer_response_rows)
print("Reviewer-requested fusion comparators (pooled, recording-grouped OOF)")
display(reviewer_response_df.round(4))

reference_scores = {
    "Nested tri-branch gate (published)": oof_tri_gate,
    "Ranked prior (strongest single branch)": ranked_prior_oof,
}
reviewer_response_deltas = []
for variant_name, variant_scores in new_variant_scores.items():
    for reference_name, ref_scores in reference_scores.items():
        lower, median, upper = paired_cluster_bootstrap_delta(variant_scores, ref_scores, repetitions=250, seed=SEED + 60)
        observed = macro_auc(targets, variant_scores)[0] - macro_auc(targets, ref_scores)[0]
        reviewer_response_deltas.append({
            "comparison": f"{variant_name} - {reference_name}",
            "observed_delta": observed, "bootstrap_median": median, "ci_2.5%": lower, "ci_97.5%": upper,
        })
reviewer_response_delta_df = pd.DataFrame(reviewer_response_deltas)
print("\nPaired recording-cluster bootstrap deltas")
display(reviewer_response_delta_df.round(4))

undefined_auc_df = pd.DataFrame(undefined_auc_counts)
print("\nClasses with undefined inner-fold AUC per branch, per outer fold (R1 Q2)")
display(undefined_auc_df)

grid_rows = [
    {"kappa": k, "gamma": g, "mean_outer_auc": float(np.mean(v)), "std_outer_auc": float(np.std(v, ddof=1))}
    for (k, g), v in grid_auc.items()
]
grid_df = pd.DataFrame(grid_rows).sort_values(["kappa", "gamma"]).reset_index(drop=True)
print("\nKappa/gamma sensitivity sweep, global constants applied across all outer folds (R1 Q1)")
display(grid_df.round(4))

masking_rows = []
masked_classes = source_coverage == 0
for name, scores in {"Hard mask (published)": oof_tri_gate, "Soft floor mask": oof_soft_mask_gate}.items():
    overall_auc, _ = macro_auc(targets, scores)
    if masked_classes.any():
        masked_auc, masked_n = macro_auc(targets[:, masked_classes], scores[:, masked_classes])
    else:
        masked_auc, masked_n = np.nan, 0
    masking_rows.append({
        "variant": name, "overall_macro_auc": overall_auc,
        "source_masked_classes_auc": masked_auc, "source_masked_classes": masked_n,
    })
masking_df = pd.DataFrame(masking_rows)
print("\nMasking ablation: hard mask vs. soft floor for source-masked classes (R1 Q1)")
display(masking_df.round(4))

global_weight_df = pd.DataFrame(global_weight_records)
ensemble_weight_df = pd.DataFrame(ensemble_weight_records)
print("\nGlobal (non-classwise) fusion weights per fold")
display(global_weight_df.round(3))
print("\nGreedy ensemble-selection weights per fold")
display(ensemble_weight_df.round(3))

reviewer_response_df.to_csv("/kaggle/working/reviewer_response_fusion_baselines.csv", index=False)
reviewer_response_delta_df.to_csv("/kaggle/working/reviewer_response_paired_deltas.csv", index=False)
undefined_auc_df.to_csv("/kaggle/working/reviewer_response_undefined_auc_counts.csv", index=False)
grid_df.to_csv("/kaggle/working/reviewer_response_kappa_gamma_grid.csv", index=False)
masking_df.to_csv("/kaggle/working/reviewer_response_masking_ablation.csv", index=False)
np.savez_compressed(
    "/kaggle/working/reviewer_response_oof_predictions.npz",
    targets=targets,
    fold_ids=fold_ids,
    prior_temporal_average=new_variant_scores["Prior + temporal average"],
    global_weighted=oof_global_weight,
    best_branch=oof_best_branch,
    logistic_stack=oof_logistic_stack,
    ensemble_selection=oof_ensemble_selection,
    soft_mask_gate=oof_soft_mask_gate,
    best_branch_picks=best_branch_picks,
)
print("\nSaved reviewer-response fusion baselines, deltas, grid, masking ablation, and OOF arrays under /kaggle/working")

site_labels = windows_df.loc[sequence_rows[:, 0], "site"].to_numpy()
unique_sites = sorted(pd.unique(site_labels))
print(f"Unique sites for leave-site-out evaluation: {unique_sites} ({len(unique_sites)} sites)")

site_oof_gate = np.full_like(oof_source, np.nan)
site_oof_prior = np.full_like(oof_source, np.nan)
site_fold_records = []

leave_site_out_started = time.time()
for site_index, held_out_site in enumerate(unique_sites):
    validation_files = np.flatnonzero(site_labels == held_out_site)
    train_files = np.flatnonzero(site_labels != held_out_site)
    if len(validation_files) == 0 or len(train_files) < 10:
        print(f"Skipping site {held_out_site}: train={len(train_files)}, validation={len(validation_files)}")
        continue

    site_seed = SEED + 3000 + site_index
    torch.manual_seed(site_seed)
    torch.cuda.manual_seed_all(site_seed)
    model, train_hybrid, validation_hybrid, validation_prior = train_hybrid_fold(
        train_files, validation_files, fold_number=3000 + site_index
    )
    weights, _ = nested_tri_branch_weights(train_files, 3000 + site_index)

    rows = grouped_rows(validation_files)
    validation_source = sequence_source_logits[validation_files].numpy()
    validation_prior_np = validation_prior.numpy()
    validation_hybrid_np = validation_hybrid.numpy()
    ranked_site = np.stack([
        rank_normalize(validation_source.reshape(-1, NUM_CLASSES)),
        rank_normalize(validation_prior_np.reshape(-1, NUM_CLASSES)),
        rank_normalize(validation_hybrid_np.reshape(-1, NUM_CLASSES)),
    ])
    site_scores = (weights[:, None, :] * ranked_site).sum(axis=0)
    site_oof_gate[rows] = site_scores
    site_oof_prior[rows] = rank_normalize(validation_prior_np.reshape(-1, NUM_CLASSES))

    gate_auc, gate_classes = macro_auc(targets[rows], site_scores)
    prior_auc, _ = macro_auc(targets[rows], site_oof_prior[rows])
    site_fold_records.append({
        "held_out_site": held_out_site, "train_files": len(train_files), "validation_files": len(validation_files),
        "classes": gate_classes, "site_held_out_gate_auc": gate_auc, "site_held_out_prior_auc": prior_auc,
    })
    print(f"site={held_out_site} gate_auc={gate_auc:.4f} prior_auc={prior_auc:.4f} files={len(validation_files)} classes={gate_classes}")
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

site_results_df = pd.DataFrame(site_fold_records)
print(f"\nCompleted leave-site-out evaluation in {time.time() - leave_site_out_started:.1f}s")
display(site_results_df.round(4))

evaluated_site_rows = ~np.isnan(site_oof_gate).any(axis=1)
if evaluated_site_rows.any():
    pooled_site_gate_auc, pooled_site_classes = macro_auc(targets[evaluated_site_rows], site_oof_gate[evaluated_site_rows])
    pooled_site_prior_auc, _ = macro_auc(targets[evaluated_site_rows], site_oof_prior[evaluated_site_rows])
    print(f"Pooled leave-site-out gate AUC={pooled_site_gate_auc:.4f}, prior AUC={pooled_site_prior_auc:.4f} over {pooled_site_classes} classes")
    print(f"Within-distribution recording-grouped gate AUC for comparison={macro_auc(targets, oof_tri_gate)[0]:.4f}")
else:
    print("No site had enough recordings on both sides of the split; leave-site-out could not be evaluated.")

site_results_df.to_csv("/kaggle/working/reviewer_response_leave_site_out.csv", index=False)
np.savez_compressed(
    "/kaggle/working/reviewer_response_leave_site_out_oof.npz",
    targets=targets, site_labels=site_labels, gate=site_oof_gate, prior=site_oof_prior,
)

LATENCY_WARMUP = 5
LATENCY_REPEATS = 50


def measure_cpu_latency(module, example_inputs, warmup=LATENCY_WARMUP, repeats=LATENCY_REPEATS):
    module = module.to("cpu").eval()
    example_inputs = tuple(x.to("cpu") for x in example_inputs)
    with torch.inference_mode():
        for _ in range(warmup):
            module(*example_inputs)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            module(*example_inputs)
            timings.append(time.perf_counter() - started)
    timings = np.array(timings)
    return {
        "median_ms": float(np.median(timings) * 1000),
        "p95_ms": float(np.percentile(timings, 95) * 1000),
        "worst_ms": float(timings.max() * 1000),
    }


def count_flops(module, example_inputs):
    module = module.to("cpu").eval()
    example_inputs = tuple(x.to("cpu") for x in example_inputs)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.inference_mode():
            counter = FlopCounterMode(display=False)
            with counter:
                module(*example_inputs)
            return int(counter.get_total_flops())
    except Exception as error:
        parameter_count = sum(p.numel() for p in module.parameters())
        print(f"FlopCounterMode unavailable ({error!r}); falling back to 2x parameter count as a FLOPs approximation.")
        return 2 * parameter_count


single_window_mel = soundscape_mels[:1].float()
source_cpu_latency = measure_cpu_latency(source_model, (single_window_mel,))
source_flops = count_flops(source_model, (single_window_mel,))

temporal_probe = TemporalHybridModel(num_sites=int(windows_df["site_index"].max()) + 1)
temporal_inputs = (
    sequence_embeddings[:1].float(),
    sequence_source_logits[:1].float(),
    torch.zeros(1, 12, NUM_CLASSES),
    sequence_sites[:1],
    sequence_hours[:1],
)
temporal_cpu_latency = measure_cpu_latency(temporal_probe, temporal_inputs)
temporal_flops = count_flops(temporal_probe, temporal_inputs)

windows_per_recording = 12
cpu_report_df = pd.DataFrame([
    {"stage": "Acoustic source CNN, per 5-s window", **source_cpu_latency, "flops": source_flops},
    {"stage": "Temporal hybrid, per 60-s recording (12 windows)", **temporal_cpu_latency, "flops": temporal_flops},
])
total_ms_per_recording = source_cpu_latency["median_ms"] * windows_per_recording + temporal_cpu_latency["median_ms"]
recordings_per_90min = (90 * 60) / (total_ms_per_recording / 1000)

print(f"Measured on: {platform.processor() or platform.machine()}, Python {platform.python_version()}, PyTorch {torch.__version__}, device=cpu")
display(cpu_report_df.round(4))
print(f"Estimated median latency per 60-s recording = {total_ms_per_recording:.2f} ms (source CNN x12 windows + temporal hybrid; prior lookup and rank-weighted sum are negligible table/array operations).")
print(f"At this measured latency, one CPU thread processes about {recordings_per_90min:.0f} sixty-second recordings within the competition's 90-minute budget.")

cpu_report_df.to_csv("/kaggle/working/reviewer_response_cpu_latency_flops.csv", index=False)

all_active_labels = set()
for labels in all_windows_df["label_list"]:
    all_active_labels.update(labels)
retained_active_labels = {class_names[index] for index in np.flatnonzero(active_mask)}
dropped_by_exclusion = sorted(all_active_labels - retained_active_labels)
print("Effect of excluding the 7 partial (<12-window) files:")
print(f"  active classes before complete-file filtering = {len(all_active_labels)}")
print(f"  active classes after complete-file filtering (evaluated here) = {len(retained_active_labels)}")
print(f"  classes whose only positive evidence lived in the excluded files ({len(dropped_by_exclusion)}): {dropped_by_exclusion}")

fig3_systems = {
    "Rank-normalized focal source CNN": ranked_source_oof,
    "MobileNetV3-Small": architecture_ranked_oof["MobileNetV3-Small"],
    "ResNet-18": architecture_ranked_oof["ResNet-18"],
    "EfficientNet-B0": architecture_ranked_oof["EfficientNet-B0"],
    "Uniform tri-branch rank average": uniform_tri_oof,
    "Rank-normalized temporal hybrid": ranked_hybrid_oof,
    "Rank-normalized site-hour prior": ranked_prior_oof,
    "Proposed tri-branch reliability gate": oof_tri_gate,
}
fig3_rows = []
for name, scores in fig3_systems.items():
    auc, classes = macro_auc(targets, scores)
    lower, upper = cluster_bootstrap_auc(scores, repetitions=500, seed=SEED + 70)
    fig3_rows.append({"system": name, "macro_oof_auc": auc, "classes": classes, "ci_2.5%": lower, "ci_97.5%": upper})
fig3_ci_df = pd.DataFrame(fig3_rows)
print("\nSingle-arm bootstrap 95% CI for every bar in Fig. 3 (use these to draw the requested error whiskers)")
display(fig3_ci_df.round(4))
fig3_ci_df.to_csv("/kaggle/working/reviewer_response_fig3_error_bars.csv", index=False)