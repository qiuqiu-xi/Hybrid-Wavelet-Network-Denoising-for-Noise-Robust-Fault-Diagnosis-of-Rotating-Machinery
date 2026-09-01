# Reproducibility Notes

## Confirmed

- The staged HDF5 files are byte-identical to the canonical copies in the
  historical noise-generation directories.
- Gaussian test-noise generation uses NumPy seed `42` in the historical
  notebooks.
- The supplied `.pkl` files are PyTorch state dictionaries rather than complete
  serialized training environments.
- The result-to-round mapping is recorded in `file_manifest.csv`.

## Known limitations

1. The code that converts official raw CWRU and SUFD files into the clean HDF5
   datasets was not found in the audited directories. Class mapping, 1024-point
   segmentation, and train/validation/test split creation therefore cannot be
   regenerated from the official downloads using this package alone.
2. Training seeds are not consistently fixed. Re-evaluation of a supplied
   checkpoint is supported, but a fresh training run may not reproduce the
   exact reported number.
3. The SUFD noise notebook currently contains a mutable `SNR_DB` setting. It
   should be parameterized before being presented as a general generation tool.
4. Some historical confusion-matrix cells plot the final mini-batch matrix
   instead of the accumulated full-test-set matrix. Legacy JPG outputs were not
   copied into the primary research-artifact set.
5. Repeated test-loader evaluations in historical notebooks are not independent
   retraining runs.

## Publication tasks still required

- Add a parameterized preprocessing/noise-generation entry point.
- Record exact package and CUDA versions from the original environment if they
  can still be recovered.
- Add unified seeds and deterministic settings for future retraining.
- Confirm dataset redistribution permissions.
- Regenerate full-test-set confusion matrices from the staged checkpoints.
- Add repository DOI and final citation metadata after deposition.

