# RF Fingerprinting Dataset & Preprocessing Pipeline

This repository holds an RF dataset recorded in the **Signal Metadata Format (SigMF)**, and the code used to access those recordings for understanding signal processing flow and RF device fingerprinting.

---
## Experiment Overview

### Devices
Data was collected across seven software-defined radios (SDRs) in total:

| Model | Device Identifiers |
| :--- | :--- |
| **USRP B210** | `B210_1`, `B210_2`, `B210_3` |
| **USRP N210** | `N210_1` |
| **USRP X300** | `X300_1`, `X300_2`, `X300_3` |

### Instances
Each instance from `1` to `34` corresponds to a separate recording session. Data was collected over 20 different days between October 21 and November 25, 2025, introducing natural variations across instances (such as temperature drift and minor setup modifications).

### Experiment Setup & Data Collection
The testbed was deployed inside an indoor laboratory environment designed to capture a realistic wireless channel characterized by multi-path effects. 

Our data collection architecture relies on a specialized setup where the underlying signal flow and execution are built entirely using the [GNU Radio](https://www.gnuradio.org/) framework, ensuring deterministic sampling and standard-compliant signal handling. The base implementation adapts the `rx_ofdm.grc` template from the [Official GNU Radio Repository](https://github.com/gnuradio/gnuradio/tree/master/gr-digital/examples/ofdm), utilizing custom blocks to record data at every stage of the signal processing flow.

The data collection pipeline was engineered to isolate and capture distinct hardware impairments unique to each device. The process follows these six steps:

1. **Transmission:** Signals are generated and transmitted via the target USRP devices.

2. **Reception:** Data is captured using a dedicated `USRP B210` receiver connected to the host PC via a `USB 3.0` interface.

3. **Multi-Session Capture:** Recordings are spread over multiple weeks to account for environmental variations. The exact date and session details are preserved in the metadata.

4. **Pre-Processing & Validation:** Frame headers and cyclic prefixes are stripped. The payload is validated via an appended 4-byte CRC checksum; frames failing the CRC check are automatically discarded.

5. **IQ Imbalance Estimation:** Amplitude and phase mismatches are jointly estimated using frequency-offset-corrected data before equalization. Outliers from the IQ imbalance estimation are filtered out, and their positions are logged externally.

6. **Metadata Tagging:** Captured IQ samples, carrier frequency offsets, channel state information (CSI), transmitted bytes, and estimated IQ imbalance parameters are serialized and paired with SigMF metadata files to ensure strict versioning and reproducibility.



<!-- Rephrase the dl to signal processing techniques -->
<!-- Info about the experiment setup -->
<!-- Data collection steps at the top -->
<!-- References to the official GNURadio website for the signal flow saying we collected data using that setup -->

---

<!-- 

Alongside the raw I/Q captures, we extract a handful of physical-layer features per device: amplitude mismatch ($\epsilon$), phase mismatch ($\tan\Delta\phi$), and both coarse and fine frequency offset (CFO/FFO). These come from multiple hardware transceivers, so the dataset can be used to study how fingerprinting performance holds up across different radio hardware.

--- -->
## Dataset Structure

Recordings are organized by capture session first, then by device. A "session" here just means one sitting where we recorded from all the devices - we call these **instances**.

```text
sigmf_dataset/
└── [instance_id]/                                  # e.g., 1, 2, 3, 4, 5, etc.
    ├── [device_id]/                                # e.g., B210_1, N210_1, X300_1
    │   ├── dataset_[instance_id]_[device].sigmf-meta  # SigMF metadata file
    │   ├── dataset_[instance_id]_[device].sigmf-data  # Raw I/Q binary file
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

## More Information about the files

> **Frame format:** These files contain only the payload section of each frame, with every frame consisting of 192 samples. The cyclic prefix has already been stripped, and every included frame passed CRC.

#### `dataset_[instance_id]_[device].sigmf-meta`

The SigMF metadata file is stored as JSON. When we execute `sigmf.fromfile()`, this file is read first. This file describes the datatype (e.g. `cf32` for complex 32-bit float), center frequency, capture date, capture session, transmitter and receiver hardware information, channel condition and other capture annotations needed to correctly interpret the paired `.sigmf-data` file.   

#### `dataset_[instance_id]_[device].sigmf-data`
This file consists of raw binary I/Q data. It is just a flat stream of complex samples. This file is not interpretable on its own and needs the matching `.sigmf-meta` file alongside it (same folder, same base filename) so `sigmf` knows how to decode the bytes into samples.


#### `fine_cfo.npy`

Fine carrier frequency offset estimates calculated using the Schmidl & Cox synchronization algorithm.

#### `ffo_compensation_sig.npy`

The signal used to compensate for fine frequency offset. This is an intermediate array from the FFO estimation and correction step. This signal can be multiplied with the raw capture to correct the fine frequency offset.


#### `coarse_cfo.npy`

Coarse carrier frequency offset estimates, one value per frame. These are integer offset estimated using the synchronization symbol. Since most of the offset is corrected in fine frequency offset step, it has zero value for all the devices.

#### `fd_channel_taps.npy`

Frequency-domain channel taps used for equalization process. It contains per-frame channel estimate used to correct for multipath/channel distortion before demodulation. Not used directly in training process, but useful if you want to inspect channel conditions per device/instance.

#### `data_bytes.npy`

The demodulated payload bytes recovered from each frame after the receiver chain (synchronization, equalization, demodulation) runs. Useful if you want to sanity-check that a recording actually decoded correctly, rather than just looking at raw I/Q.


#### `iq_epsilon.npy` and `iq_tan_delta_phi.npy`

The extracted amplitude mismatch parameter (`iq_epsilon.npy`) and phase mismatch parameter (`iq_tan_delta_phi.npy`) per frame which tends to be a hardware-specific characteristic of a given transmitter's front end. They are RF-fingerprinting features used alongside the raw I/Q and extracted using the `data_bytes` and frequency offset compensated time-domain I/Q data. 


#### `outlier_position.npy`

A boolean mask, one entry per frame, flagging frames that were identified as outliers (based on the amplitude/phase mismatch parameters) and should be excluded. `load_and_preprocess()` uses this to drop the corresponding frames from both the raw I/Q and the parameter arrays before they're concatenated.

#### `outlier_removed_ffo.npy`

The fine frequency offset values with outlier frames already stripped out. This is the "clean" FFO array actually loaded into `X_device_parameter`, so its length should line up with the outlier-filtered I/Q frame count rather than the original raw frame count.




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
This should output 

```
(1147968,) complex64

```

From there, `np.real(samples)` and `np.imag(samples)` split out the I and Q components if you need them separately. In our pipeline, each device's stream gets chopped into fixed-length frames of 192 samples (`FRAME_LEN`) before anything else happens.

---

## Full Preprocessing Pipeline

<!-- Upload the code and remove the code from the documentation and refer to filename for the description-->
<!-- CFO/FFO Extracted from the ofdm blocks -->

The [`data_preprocessing.ipynb`](https://github.com/bimal-tech/Dataset/tree/master) consists of `load_and_preprocess()` function that we actually use to build a training-ready dataset. It walks through every device/instance combination, loads the I/Q data, drops flagged outlier frames, pulls in the pre-computed parameter features (epsilon, tan-delta-phi, FFO), and optionally smooths those parameters with a moving average. Everything gets normalized and labeled at the end.


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
| `scalers_raw_iq` | `StandardScaler` | Fitted I/Q scaler, reusable on a validation or test split |
| `scalers_parameter` | `StandardScaler` | Fitted scaler for the parameter features, also reusable |

