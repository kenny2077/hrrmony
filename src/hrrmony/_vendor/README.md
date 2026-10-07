# Vendored code

`infer_rvc_python/` is a copy of [R3gm/infer_rvc_python](https://github.com/R3gm/infer_rvc_python), MIT license (see `infer_rvc_python/LICENSE`). It is vendored so that `hrrmony` can be installed together with `audio-separator`: the upstream package pins `torchcrepe==0.0.20`, which pins `librosa==0.9.1`, and that conflicts with `audio-separator`'s `librosa>=0.10`.

Local changes:
- imports rewritten to `hrrmony._vendor.infer_rvc_python`
- `torchcrepe` is imported lazily, only when the `crepe` pitch method is used (hrrmony uses `rmvpe`)
