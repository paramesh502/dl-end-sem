import time
import os
import ctypes
import ctypes.wintypes

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import spectrogram


# ============================================================
# I1 WEEK-1 REFERENCE RUN
# Satellite Telemetry Anomaly Detection
# 2-D Spectrogram Autoencoder
# ============================================================

SEED = 22
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Run configuration
# ------------------------------------------------------------

torch.set_num_threads(6)

NUM_WORKERS = 0
MAX_EPOCHS = 30
MAX_SECONDS = 30 * 60


# ------------------------------------------------------------
# 1. Load OPSSAT data
# ------------------------------------------------------------

print("Loading OPSSAT data...")

df = pd.read_csv("../segments.csv")
dev_df = pd.read_csv("../OPSSAT_nominal_dev_segments.csv")

dev_ids = set(dev_df["segment"].astype(str))

nominal_train = df[
    (df["train"].astype(str) == "1") &
    (df["anomaly"].astype(str) == "0")
].copy()

nominal_train["segment"] = nominal_train["segment"].astype(str)

train_ids = set(nominal_train["segment"]) - dev_ids

print("Nominal fitting segments:", len(train_ids))
print("Nominal development segments:", len(dev_ids))


# ------------------------------------------------------------
# 2. Resample every segment to 256 samples
# ------------------------------------------------------------

def resample_to_256(values):

    values = np.asarray(values, dtype=np.float32)

    if len(values) == 256:
        return values

    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, 256)

    return np.interp(
        new_x,
        old_x,
        values
    ).astype(np.float32)


def make_segments(segment_ids):

    result = []

    for sid in segment_ids:

        part = nominal_train[
            nominal_train["segment"] == sid
        ]

        values = part["value"].astype(float).to_numpy()

        if len(values) < 2:
            continue

        values = resample_to_256(values)

        result.append(values)

    return np.stack(result)


print("Preparing fitting data...")

X_train_raw = make_segments(train_ids)

print("Preparing development data...")

X_val_raw = make_segments(dev_ids)

print("Train shape:", X_train_raw.shape)
print("Validation shape:", X_val_raw.shape)


# ------------------------------------------------------------
# 3. Training-only standardisation
# ------------------------------------------------------------

train_mean = X_train_raw.mean()
train_std = X_train_raw.std()

if train_std == 0:
    train_std = 1.0

X_train = (X_train_raw - train_mean) / train_std
X_val = (X_val_raw - train_mean) / train_std


# ------------------------------------------------------------
# 4. Convert telemetry into 2-D spectrograms
# ------------------------------------------------------------

def to_spectrogram(x):

    frequencies, times, Sxx = spectrogram(
        x,
        nperseg=64,
        noverlap=32,
        nfft=64,
        scaling="density",
        mode="magnitude"
    )

    Sxx = np.log1p(Sxx)

    return Sxx.astype(np.float32)


X_train_img = np.stack(
    [to_spectrogram(x) for x in X_train]
)

X_val_img = np.stack(
    [to_spectrogram(x) for x in X_val]
)

print("Spectrogram shape:", X_train_img.shape)

X_train_img = X_train_img[:, None, :, :]
X_val_img = X_val_img[:, None, :, :]


# ------------------------------------------------------------
# 5. Dataset
# ------------------------------------------------------------

class SpectrogramDataset(Dataset):

    def __init__(self, x):
        self.x = torch.tensor(
            x,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


train_dataset = SpectrogramDataset(
    X_train_img
)

val_dataset = SpectrogramDataset(
    X_val_img
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ------------------------------------------------------------
# 6. 2-D convolutional autoencoder
# ------------------------------------------------------------

class SpectrogramAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        z = self.encoder(x)

        return self.decoder(z)


model = SpectrogramAutoencoder()

parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", parameter_count)


# ------------------------------------------------------------
# 7. Training setup
# ------------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()

best_val_loss = float("inf")

best_state = None

epochs_without_improvement = 0

history = []   # (epoch, train_loss, val_loss) for the loss curve


# ------------------------------------------------------------
# 8. Start timed run
# ------------------------------------------------------------

train_start = time.perf_counter()

cpu_start = time.process_time()

print()
print("========================================")
print("STARTING TIMED WEEK-1 RUN")
print("========================================")

print("CPU threads:", torch.get_num_threads())

print("DataLoader workers:", NUM_WORKERS)

print("Maximum epochs:", MAX_EPOCHS)

print("Maximum elapsed time: 30 minutes")

print()


# ------------------------------------------------------------
# 9. Training loop
# ------------------------------------------------------------

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    elapsed = (
        time.perf_counter()
        - train_start
    )

    if elapsed >= MAX_SECONDS:

        print(
            "30-minute elapsed limit reached."
        )

        break


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0

    train_count = 0


    for batch in train_loader:

        optimizer.zero_grad()

        output = model(batch)

        loss = criterion(
            output,
            batch
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item()
            * len(batch)
        )

        train_count += len(batch)


        if (
            time.perf_counter()
            - train_start
            >= MAX_SECONDS
        ):

            break


    train_loss = (
        train_loss_sum
        / max(train_count, 1)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0

    val_count = 0


    with torch.no_grad():

        for batch in val_loader:

            output = model(batch)

            loss = criterion(
                output,
                batch
            )

            val_loss_sum += (
                loss.item()
                * len(batch)
            )

            val_count += len(batch)


    val_loss = (
        val_loss_sum
        / max(val_count, 1)
    )


    elapsed = (
        time.perf_counter()
        - train_start
    )


    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Elapsed: {elapsed/60:.2f} min"
    )


    history.append((epoch, train_loss, val_loss))


    # --------------------------------------------------------
    # Save best validation model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if epochs_without_improvement >= 5:

        print(
            "Early stopping: "
            "5 validation checks without improvement."
        )

        break


# ------------------------------------------------------------
# 10. Restore best model
# ------------------------------------------------------------

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ------------------------------------------------------------
# 11. Timing results
# ------------------------------------------------------------

elapsed_training = (
    time.perf_counter()
    - train_start
)

cpu_training = (
    time.process_time()
    - cpu_start
)


print()
print("========================================")
print("TRAINING FINISHED")
print("========================================")

print(
    f"Elapsed training time: "
    f"{elapsed_training:.2f} seconds"
)

print(
    f"Aggregate main-process CPU time: "
    f"{cpu_training:.2f} seconds"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)


# ------------------------------------------------------------
# 12. Measure peak Python memory correctly
# ------------------------------------------------------------

class PROCESS_MEMORY_COUNTERS(
    ctypes.Structure
):

    _fields_ = [

        (
            "cb",
            ctypes.wintypes.DWORD
        ),

        (
            "PageFaultCount",
            ctypes.wintypes.DWORD
        ),

        (
            "PeakWorkingSetSize",
            ctypes.c_size_t
        ),

        (
            "WorkingSetSize",
            ctypes.c_size_t
        ),

        (
            "QuotaPeakPagedPoolUsage",
            ctypes.c_size_t
        ),

        (
            "QuotaPagedPoolUsage",
            ctypes.c_size_t
        ),

        (
            "QuotaPeakNonPagedPoolUsage",
            ctypes.c_size_t
        ),

        (
            "QuotaNonPagedPoolUsage",
            ctypes.c_size_t
        ),

        (
            "PagefileUsage",
            ctypes.c_size_t
        ),

        (
            "PeakPagefileUsage",
            ctypes.c_size_t
        )
    ]


pmc = PROCESS_MEMORY_COUNTERS()

pmc.cb = ctypes.sizeof(pmc)


psapi = ctypes.WinDLL(
    "Psapi.dll"
)

kernel32 = ctypes.WinDLL(
    "Kernel32.dll"
)


psapi.GetProcessMemoryInfo.argtypes = [

    ctypes.wintypes.HANDLE,

    ctypes.POINTER(
        PROCESS_MEMORY_COUNTERS
    ),

    ctypes.wintypes.DWORD
]


psapi.GetProcessMemoryInfo.restype = (
    ctypes.wintypes.BOOL
)


handle = (
    kernel32.GetCurrentProcess()
)


success = (
    psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(pmc),
        ctypes.sizeof(pmc)
    )
)


if success:

    peak_memory_mb = (
        pmc.PeakWorkingSetSize
        / (1024 ** 2)
    )

    print(
        f"Peak Python working-set memory: "
        f"{peak_memory_mb:.2f} MB"
    )

else:

    print(
        "Could not measure peak memory."
    )


# ------------------------------------------------------------
# 13. Timed unscored inference
# ------------------------------------------------------------

print()

print(
    "Starting timed unscored inference..."
)


inference_start = time.perf_counter()


model.eval()


with torch.no_grad():

    for batch in val_loader:

        _ = model(batch)


inference_time = (
    time.perf_counter()
    - inference_start
)


print(
    f"Timed inference over "
    f"validation-shaped data: "
    f"{inference_time:.2f} seconds"
)


# ------------------------------------------------------------
# 13b. [ADDED] Start / end losses for the readiness note
# ------------------------------------------------------------

print()
print(
    f"Train loss  epoch 1 -> epoch {history[-1][0]}: "
    f"{history[0][1]:.6f} -> {history[-1][1]:.6f}"
)
print(
    f"Val loss    epoch 1 -> epoch {history[-1][0]}: "
    f"{history[0][2]:.6f} -> {history[-1][2]:.6f}"
)
print(f"Epochs completed: {history[-1][0]}")
print(f"Optimisation updates: {history[-1][0] * len(train_loader)}")


# ------------------------------------------------------------
# 13c. [ADDED] Timed, UNSCORED inference over test-shaped data
#      (OPSSAT test fragments, train == 0). Only the "value"
#      column is used; the anomaly label is never read.
# ------------------------------------------------------------

test_df = df[df["train"].astype(str) == "0"]

test_segments = []

for sid, part in test_df.groupby("segment"):

    values = part["value"].astype(float).to_numpy()

    if len(values) < 2:
        continue

    test_segments.append(resample_to_256(values))

X_test_raw = np.stack(test_segments)

X_test = (X_test_raw - train_mean) / train_std

X_test_img = np.stack(
    [to_spectrogram(x) for x in X_test]
)[:, None, :, :]

test_loader = DataLoader(
    SpectrogramDataset(X_test_img),
    batch_size=32,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print()
print("Test-shaped fragments (inputs only):", len(X_test_raw))

test_inf_start = time.perf_counter()

model.eval()

with torch.no_grad():

    for batch in test_loader:

        _ = model(batch)

test_inf_time = (
    time.perf_counter()
    - test_inf_start
)

print(
    f"Timed inference over {len(X_test_raw)} "
    f"resampled test fragments: {test_inf_time:.2f} seconds"
)


# ------------------------------------------------------------
# 13d. [ADDED] Save the trained model + loss curve
# ------------------------------------------------------------

torch.save(
    {
        "state_dict": model.state_dict(),
        "train_mean": float(train_mean),
        "train_std": float(train_std),
    },
    "model_week1.pt"
)

pd.DataFrame(
    history,
    columns=["epoch", "train_loss", "val_loss"]
).to_csv("week1_loss_history.csv", index=False)

print("Saved model_week1.pt and week1_loss_history.csv")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h[0] for h in history]
    plt.figure(figsize=(6, 3.5))
    plt.plot(ep, [h[1] for h in history], label="train")
    plt.plot(ep, [h[2] for h in history], label="validation")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Week-1 spectrogram autoencoder")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week1_loss_curve.png", dpi=150)
    print("Saved week1_loss_curve.png")
except Exception as e:
    print("Could not draw loss curve:", e)


# ------------------------------------------------------------
# 14. Final note
# ------------------------------------------------------------

print()

print("IMPORTANT:")

print(
    "No test labels or test metrics were used."
)

print(
    "Week-1 reference run complete."
)