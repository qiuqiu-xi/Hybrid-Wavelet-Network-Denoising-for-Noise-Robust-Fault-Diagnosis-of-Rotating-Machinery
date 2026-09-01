# Hybrid Wavelet-Network Denoising: Research Artifacts

This directory stages the research artifacts used for the manuscript
"Hybrid Wavelet-Network Denoising for Noise-Robust Fault Diagnosis of
Rotating Machinery."

## Contents

- `code/cwru/`: selected CWRU notebooks for the six reported noise conditions.
- `code/sufd/`: selected SUFD notebooks for the three reported conditions.
- `code/noise_generation/`: historical Gaussian-noise generation notebooks.
- `data/processed/`: deduplicated HDF5 datasets used by the selected notebooks.
- `models/`: model state dictionaries associated with the reported results.
- `results/`: authoritative CSV tables corresponding to the manuscript.
- `code/plotting/`: reproducible accuracy, gain, interaction, and radar plots.
- `results/figures/`: corrected publication figures derived from the staged data.
- `docs/file_manifest.csv`: source paths, hashes, and experiment mapping.
- `docs/checksums.sha256`: SHA-256 checksums for repository files.

The workbooks under `docs/legacy_result_workbooks/` are retained only for
provenance. In particular, the legacy CWRU workbook still contains the
superseded 15 dB Hybrid value `0.9534`. The corrected plotting source is
`results/plot_data_cwru.xlsx`, and its tabular counterpart is
`results/cwru_results.csv`.

## Reported experiment mapping

For CWRU, the clean, 20 dB, 15 dB, 5 dB, and 0 dB artifacts come from
`ROUND_4`. The 10 dB artifacts come from `ROUND_5`.

For SUFD, the clean artifacts come from `Round_2/No_noisy`; all 15 dB
artifacts come from `Round_1/SNR=15`. At 20 dB, the Basic and Model-denoising
artifacts come from `Round_1/SNR=20`, whereas the Wavelet and Hybrid artifacts
come from `Round_2/SNR=20`.

The manuscript's CWRU Hybrid result at 15 dB was corrected from `0.9534` to
the traceable value `0.9512` (notebook output: `0.951182`).

## Reproducibility scope

The staged files support evaluation of the supplied checkpoints and partial
retraining from the processed HDF5 files. They do not yet provide a complete
official-raw-data-to-HDF5 pipeline. Training notebooks also do not consistently
set all Python, NumPy, PyTorch, CUDA, and DataLoader random seeds. See
`docs/REPRODUCIBILITY_NOTES.md` before publishing the repository.

## Data publication

The HDF5 and checkpoint files are staged locally. Before public release,
confirm that the CWRU and SUFD licenses permit redistribution of processed
copies. A practical release is to host code and documentation in GitHub and
archive large data/model files in a DOI-bearing research-data repository.
