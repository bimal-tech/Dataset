# SigMF RF Fingerprinting Dataset & Preprocessing Pipeline

This repo holds an RF dataset recorded in the **Signal Metadata Format (SigMF)**, plus the code used to turn those recordings into training data for deep-learning-based device fingerprinting.

Alongside the raw I/Q captures, we extract a handful of physical-layer features per device: amplitude mismatch ($\epsilon$), phase mismatch ($\tan\Delta\phi$), and both coarse and fractional frequency offset (CFO/FFO). These come from multiple hardware transceivers, so the dataset can be used to study how fingerprinting performance holds up across different radio hardware.

---
## Dataset Structure

Recordings are organized by capture session first, then by device. A "session" here just means one sitting where we recorded from all the devices - we call these **instances**.

```text
sigmf_dataset/
└── [instance_id]/                                  # e.g., 1, 2, 3, 4, 5, etc.
    ├── [device_id]/                                # e.g., B210_1, N210_1, X300_1
    │   ├── dataset_[instance]_[device].sigmf-meta  # SigMF metadata file
    │   ├── dataset_[instance]_[device].sigmf-data  # Raw I/Q binary file
    │   ├── coarse_cfo.npy                          # Coarse carrier frequency offset
    │   ├── fine_cfo.npy                             # Fine carrier frequency offset
    │   ├── ffo_compensation_sig.npy                 # FFO compensating signal
    │   ├── data_bytes.npy                           # Demodulated data bytes
    │   ├── fd_channel_taps.npy                      # Frequency-domain channel taps used for equalization
    │   ├── iq_epsilon.npy                           # Extracted amplitude mismatch parameter
    │   ├── iq_tan_delta_phi.npy                     # Extracted phase mismatch parameter
    │   ├── outlier_position.npy                     # Boolean mask flagging outlier frames
    │   └── outlier_removed_ffo.npy                  # FFO values with outliers stripped out
```

---

## Devices

Seven radios in total: three USRP B210s, one USRP N210, and three USRP X300s. They're labeled `B210_1`, `B210_2`, `B210_3`, `N210_1`, `X300_1`, `X300_2`, and `X300_3`.

---

## Instances

Each instance from `1` to `34` corresponds to a separate recording session. We collected data over 20 different days between October 21 and November 25, 2025, so there's some natural variation across instances (temperature drift, slightly different setups, etc.).

---

## Setup

You'll need a few packages to read the SigMF files and run the preprocessing:

```bash
pip install sigmf numpy scipy scikit-learn
```

- `sigmf` reads the `.sigmf-meta` / `.sigmf-data` pairs and hands back the complex I/Q samples
- `numpy` for the array work
- `scipy` for the moving-average smoothing on the parameter features
- `scikit-learn` for `StandardScaler`, used during normalization

---

## Reading a SigMF Recording

Each capture is stored as a pair of files: a `.sigmf-meta` JSON file (sample rate, datatype, capture details) and a `.sigmf-data` binary file with the actual samples. You only need to point at the `.sigmf-meta` file — as long as both files live in the same folder and share the same base name, `sigmf.fromfile()` will find the data file on its own.

```python
import sigmf
from pathlib import Path

meta_path = Path("sigmf_dataset/1/B210_1/dataset_1_B210_1.sigmf-meta")

sig = sigmf.fromfile(str(meta_path))
samples = sig.read_samples()

print(samples.shape, samples.dtype)

```
From there, `np.real(samples)` and `np.imag(samples)` split out the I and Q components if you need them separately. In our pipeline, each device's stream gets chopped into fixed-length frames of 192 samples (`FRAME_LEN`) before anything else happens.

---

## Full Preprocessing Pipeline

`load_and_preprocess()` is the function we actually use to build a training-ready dataset. It walks through every device/instance combination, loads the I/Q data, drops flagged outlier frames, pulls in the pre-computed parameter features (epsilon, tan-delta-phi, FFO), and optionally smooths those parameters with a moving average. Everything gets normalized and labeled at the end.

```python
import numpy as np
from pathlib import Path
import sigmf
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(
    working_instances,
    all_devices,
    normalization=True,
    scalers_raw_iq=None,
    scalers_parameter=None,
    apply_moving_average=False,
    parameter_moving_average_window_size=50,
):
    X_list_raw_iq = []
    X_list_parameter = []
    Y_list = []
    FRAME_LEN = 192

    for device_id in all_devices:
        for instance in working_instances:
            # Construct path
            base_path = Path(f"../sigmf_dataset/{instance}/{device_id}")
            filename_base = f"dataset_{instance}_{device_id}"
            meta_path = base_path / f"{filename_base}.sigmf-meta"

            epsilon_path = base_path / f"iq_epsilon.npy"
            tan_delta_phi_path = base_path / f"iq_tan_delta_phi.npy"
            ffo_path = base_path / f"outlier_removed_ffo.npy"

            outlier_path = base_path / f"outlier_position.npy"
            if not meta_path.exists():
                continue

            try:
                # Load the SigMF recording and read the complex I/Q samples
                sig = sigmf.fromfile(str(meta_path))
                samples = sig.read_samples()

                num_stacks = len(samples) // FRAME_LEN
                if num_stacks == 0:
                    continue

                samples = samples[: num_stacks * FRAME_LEN]
                long_frames = samples.reshape(num_stacks, FRAME_LEN)

                outliers = np.load(outlier_path)
                cleaned_frame = np.delete(long_frames, np.where(outliers)[0], axis=0)

                X_device_raw_iq = np.stack(
                    (np.real(cleaned_frame), np.imag(cleaned_frame)), axis=1
                )
                y_device = np.full(len(cleaned_frame), device_id)

                X_device_parameter = np.stack(
                    (
                        np.load(epsilon_path),
                        np.load(tan_delta_phi_path),
                        np.load(ffo_path),
                    ),
                    axis=1,
                )

                if apply_moving_average:
                    features = uniform_filter1d(
                        X_device_parameter,
                        size=parameter_moving_average_window_size,
                        axis=0,
                    )
                else:
                    features = X_device_parameter

                X_list_raw_iq.append(X_device_raw_iq)
                X_list_parameter.append(features)
                Y_list.append(y_device)
            except Exception as e:
                print(f"Error loading {meta_path}: {e}")

    if not X_list_raw_iq:
        print("No data found.")
        return None, None, None

    # Concatenate all loaded blocks
    X_raw = np.concatenate(X_list_raw_iq, axis=0)
    X_parameter = np.vstack(X_list_parameter)
    Y_raw = np.concatenate(Y_list, axis=0)

    if normalization:
        if scalers_raw_iq:
            X_final_raw_iq, scalers_raw_iq = global_normalize_data(
                X_raw, scalers_raw_iq
            )
        else:
            X_final_raw_iq, scalers_raw_iq = global_normalize_data(X_raw)

        if scalers_parameter:
            X_final_parameter = scalers_parameter.transform(X_parameter)
        else:
            scalers_parameter = StandardScaler()
            X_final_parameter = scalers_parameter.fit_transform(X_parameter)
    else:
        X_final_raw_iq, scalers_raw_iq = X_raw, None
        X_final_parameter, scalers_parameter = X_parameter, None

    # Map device IDs to clean integer labels (0, 1, 2, ...)
    unique_labels = sorted(np.unique(Y_raw))
    label_map = {old_label: i for i, old_label in enumerate(unique_labels)}
    Y_final = np.array([label_map[y] for y in Y_raw])

    # Reshape for deep learning models: (N, Channels, Time, 1)
    X_final_raw_iq = X_final_raw_iq[..., np.newaxis]
    X_final_parameter = X_final_parameter[..., np.newaxis]

    print(
        f"Preprocessing complete. Shape: {X_final_raw_iq.shape}, Labels: {len(unique_labels)}"
    )
    print(
        f"Preprocessing complete. Shape: {X_final_parameter.shape}, Labels: {len(unique_labels)}"
    )
    return X_final_raw_iq, X_final_parameter, Y_final, scalers_raw_iq, scalers_parameter
```

### The `global_normalize_data` helper

This is the piece that actually standardizes the raw I/Q data. It flattens the I and Q channels across every frame in the batch, fits (or reuses) a `StandardScaler` on that flattened view, then reshapes everything back to the original `(N, channels, time)` layout.

```python
def global_normalize_data(X_data, scaler=None):
    N, ch, t = X_data.shape

    # Flatten to (N * window_size, 2) to treat I and Q as separate features
    # but standardize them across the entire temporal dimension.
    X_reshaped = X_data.transpose(0, 2, 1).reshape(-1, ch)

    if scaler is None:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_reshaped)
    else:
        X_norm = scaler.transform(X_reshaped)

    # Reshape back to (N, window_size, 2) -> (N, 2, window_size)
    X_final = X_norm.reshape(N, t, ch).transpose(0, 2, 1)
    return X_final, scaler
```


### Trying it out

```python
working_instances = [1, 2, 3, 4, 5]
all_devices = ["B210_1", "B210_2", "B210_3", "N210_1", "X300_1", "X300_2", "X300_3"]

X_raw_iq, X_parameter, Y, scalers_raw_iq, scalers_parameter = load_and_preprocess(
    working_instances=working_instances,
    all_devices=all_devices,
    normalization=True,
    apply_moving_average=True,
    parameter_moving_average_window_size=50,
)

print(X_raw_iq.shape)     # (N, 2, 192, 1)  -> I and Q channels x frame length
print(X_parameter.shape)  # (N, 3, 1)       -> epsilon, tan_delta_phi, ffo
print(Y.shape)            # (N,)            -> integer device labels
```

### What you get back

| Output | Shape | What it is |
|---|---|---|
| `X_final_raw_iq` | `(N, 2, 192, 1)` | Raw I/Q frames — channel 0 is the real part, channel 1 is the imaginary part |
| `X_final_parameter` | `(N, 3, 1)` | The three physical-layer features: amplitude mismatch, phase mismatch, FFO |
| `Y_final` | `(N,)` | Integer device labels, 0-indexed |
| `scalers_raw_iq` | `StandardScaler` | Fitted I/Q scaler(s), reusable on a validation or test split |
| `scalers_parameter` | `StandardScaler` | Fitted scaler for the parameter features, also reusable |

