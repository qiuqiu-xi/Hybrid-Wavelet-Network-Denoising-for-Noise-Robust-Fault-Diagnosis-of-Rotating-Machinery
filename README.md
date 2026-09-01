# Hybrid Wavelet-Network Denoising: Research Artifacts

This repository contains the code, processed experiment inputs, trained model
state dictionaries, tabulated results, and plotting scripts associated with the
manuscript:

> Hybrid Wavelet-Network Denoising for Noise-Robust Fault Diagnosis of
> Rotating Machinery

The experiments cover the Case Western Reserve University (CWRU) bearing
dataset and the Southeast University gearbox dataset (SUFD). Four model
configurations are reported: Basic, Model Denoising, Wavelet Denoising, and
Hybrid Denoising (WND).

## Repository contents

```text
code/
  cwru/                  CWRU notebooks for clean and five noisy conditions
  sufd/                  SUFD notebooks for clean, 20 dB, and 15 dB conditions
  noise_generation/      historical Gaussian test-noise notebooks
  plotting/              scripts used to regenerate result figures
data/
  CWRU/                  compressed CWRU HDF5 experiment inputs
  SUFD/                  compressed SUFD HDF5 experiment inputs
models/
  cwru/                  CWRU state dictionaries by condition and method
  sufd/                  SUFD state dictionaries by condition and method
results/
  *.csv                  authoritative numerical results used in the manuscript
  plot_data_*.xlsx       plotting workbooks
  figures/               regenerated publication figures
docs/
  file_manifest.csv      artifact-to-experiment mapping and SHA-256 metadata
  checksums.sha256       checksums of the original staged artifacts
  REPRODUCIBILITY_NOTES.md
                         confirmed details, limitations, and remaining tasks
```

The authoritative result for CWRU Hybrid Denoising at 15 dB is `0.9512`
(notebook output: `0.951182`). The value `0.9534` in the legacy CWRU workbook is
superseded and is retained only for provenance.

## Data sources

The source datasets are available from their original release pages:

- CWRU Bearing Data Center:
  <https://engineering.case.edu/bearingdatacenter/download-data-file>
- Southeast University Mechanical Datasets repository:
  <https://github.com/cathysiyu/Mechanical-datasets/tree/master/gearbox>

The archives under `data/CWRU/` and `data/SUFD/` contain processed HDF5 inputs
used by the selected historical notebooks. Public access to a source dataset
does not by itself establish permission to redistribute derived copies. Users
must comply with the terms of the original dataset providers. Any future code
license for this repository will not automatically apply to these data files.

## Installation

Python 3.9.23 was used for the reported experiments. Create an isolated
environment and install the listed packages:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install openpyxl jupyterlab
```

The original package and CUDA versions were not fully preserved. Consequently,
`requirements.txt` currently records package names rather than an exact locked
environment.

## Preparing the processed data

Extract only the archive needed for the experiment. The archive and extracted
file names are standardized as follows:

| Dataset | Condition | Archive | Extracted file |
|---|---|---|---|
| CWRU | clean | `data/CWRU/cwru_clean.rar` | `cwru_clean.h5` |
| CWRU | 20 dB | `data/CWRU/cwru_snr20.rar` | `cwru_snr20.h5` |
| CWRU | 15 dB | `data/CWRU/cwru_snr15.rar` | `cwru_snr15.h5` |
| CWRU | 10 dB | `data/CWRU/cwru_snr10.rar` | `cwru_snr10.h5` |
| CWRU | 5 dB | `data/CWRU/cwru_snr5.rar` | `cwru_snr5.h5` |
| CWRU | 0 dB | `data/CWRU/cwru_snr0.rar` | `cwru_snr0.h5` |
| SUFD | clean | `data/SUFD/sufd_clean.rar` | `sufd_clean.h5` |
| SUFD | 20 dB | `data/SUFD/sufd_snr20.rar` | `sufd_snr20.h5` |
| SUFD | 15 dB | `data/SUFD/sufd_snr15.rar` | `sufd_snr15.h5` |

RAR extraction requires a compatible tool such as 7-Zip or WinRAR.

The current `docs/checksums.sha256` records the hashes and former
`data/processed/...` paths of the extracted HDF5 files, not hashes of the RAR
archives. Verify extracted contents using the filename mapping above. A future
release should add archive-level checksums for direct verification of the
downloaded RAR files.

## Running the historical notebooks

The notebooks are preserved as executed research records. They still contain
the historical data and checkpoint filenames listed below, whereas the
repository uses standardized artifact names. Before running a notebook, either
edit its `data_path` and `torch.load(...)` calls to point to the repository
files, or place renamed copies in the notebook working directory.

### Historical data filenames

| Dataset and condition | Historical filename in notebook | Repository filename |
|---|---|---|
| CWRU clean | `DataBase_12K_HP_0.h5` | `cwru_clean.h5` |
| CWRU noisy | `CWRU_test_noisy_SNR<N>.h5` | `cwru_snr<N>.h5` |
| SUFD clean | `Geerdataset_bace.h5` | `sufd_clean.h5` |
| SUFD noisy | `Geerdataset_SNR<N>.h5` | `sufd_snr<N>.h5` |

Here `<N>` is the numerical SNR value, for example `15` or `0`.

### Historical checkpoint filenames

| Method | Repository filename | Historical filename(s) |
|---|---|---|
| Basic | `basic_state_dict.pkl` | `Yourmodel.pkl`, `Bace_model.pkl`, or `Bace.pkl` |
| Model Denoising | `model_denoising_state_dict.pkl` | `Bace_dnn_model.pkl` |
| Wavelet Denoising | `wavelet_state_dict.pkl` | `Bace_wave_model.pkl` |
| Hybrid Denoising | `hybrid_state_dict.pkl` | `Dnn_wave_model.pkl` |

The supplied `.pkl` files are PyTorch state dictionaries. The matching model
class must be defined before calling `load_state_dict`.

## Regenerating the result figures

Run the plotting scripts from the repository root. The commands below use the
corrected workbooks under `results/` and write vector PDF figures to
`results/figures/`:

```bash
python code/plotting/plot_noise_accuracy.py \
  --input-dir results --output-dir results/figures \
  --workbooks plot_data_cwru.xlsx plot_data_sufd.xlsx --mode both

python code/plotting/plot_noise_gain.py \
  --input-dir results --output-dir results/figures \
  --workbooks plot_data_cwru.xlsx plot_data_sufd.xlsx --mode both

python code/plotting/plot_radar_robustness.py \
  --input-dir results --output-dir results/figures \
  --workbooks plot_data_cwru.xlsx plot_data_sufd.xlsx
```

The CSV files in `results/` are the compact, machine-readable source of the
reported accuracy values. The workbooks are retained because the current
plotting scripts also read the table layout used during figure preparation.

## Experiment provenance

- CWRU clean, 20 dB, 15 dB, 5 dB, and 0 dB artifacts come from `ROUND_4`.
- CWRU 10 dB artifacts come from `ROUND_5`.
- SUFD clean artifacts come from `Round_2/No_noisy`.
- SUFD 15 dB artifacts come from `Round_1/SNR=15`.
- At SUFD 20 dB, Basic and Model Denoising come from `Round_1/SNR=20`;
  Wavelet and Hybrid Denoising come from `Round_2/SNR=20`.

Additional artifact mappings and hashes are recorded in
`docs/file_manifest.csv`.

## Reproducibility scope and limitations

The repository supports inspection of the executed notebooks, evaluation of
the supplied checkpoints after path adjustment, partial retraining from the
processed HDF5 files, and regeneration of the reported result figures.

The following limitations remain:

1. The official-raw-data-to-HDF5 preprocessing pipeline was not found in the
   audited historical directories.
2. Training notebooks do not consistently fix all Python, NumPy, PyTorch,
   CUDA, and DataLoader random seeds. Fresh training may therefore differ from
   the reported checkpoint results.
3. The SUFD noise-generation notebook contains a mutable `SNR_DB` setting.
4. Some historical confusion-matrix cells visualize the final mini-batch
   rather than the accumulated full test set.
5. Repeated test-loader evaluations are not independent retraining runs.

See `docs/REPRODUCIBILITY_NOTES.md` for the detailed audit record.

## Citation

Until a versioned release and persistent identifier are available, cite the
repository URL and the exact commit used:

```text
Li, W., Wang, H., Li, M., Zang, Z., and Guo, J. (2026).
Hybrid Wavelet-Network Denoising for Noise-Robust Fault Diagnosis of
Rotating Machinery: Research Artifacts. GitHub.
https://github.com/qiuqiu-xi/Hybrid-Wavelet-Network-Denoising-for-Noise-Robust-Fault-Diagnosis-of-Rotating-Machinery
```

A `CITATION.cff` file and DOI should be added when the first stable release is
archived.

## License

A repository-level reuse license has not yet been assigned. Until a `LICENSE`
file is added, the repository remains publicly viewable but no additional
permission to reuse, modify, or redistribute the code is granted. Third-party
dataset terms apply independently.
