# GPU wavelet preprocessing

Install the runtime dependency in the training environment:

```bash
python -m pip install pytorch-wavelets
```

The default transform accepts `[N, 1, H, W]` and returns `[N, 4, H, W]` in
`[LL, LH, HL, HH]` order using one-level `db4` DWT. Change `wavelet` to
`haar`, `db8`, `sym5`, etc. For `level=J`, the output has `1 + 3 * J`
channels, so update `model.encoder.dim_in` accordingly.
