# CUDA RTC Bridge Package Clean Consumer

- Classification: `local-feed-clean-package-consumer-candidate`
- Managed package: `JYPPX.TensorRT.CSharp.API 4.0.0-rtc-local.20260728`
- Bridge package: `JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge 4.0.0-rtc-local.20260728`
- Repository-external PackageReference consumer: `True`
- Local-only NuGet feed: `True`
- Package bundles NVRTC / builtins: `False` / `False`
- Missing-NVRTC diagnostic: `True`
- NVRTC / Driver version: `12.9` / `12090`
- Runtime-library launch/readback/correctness: `True` / `True` / `True`
- Driver launch/readback/correctness: `True` / `True` / `True`
- Output hashes match: `True`
- Public/post-publish promotion: `False` / `False`

This proves a repository-external PackageReference consumer restored from a local-only feed, copied the packaged bridge, diagnosed absent NVRTC, and used a user-installed Toolkit for local compile and dual launch/readback. It is not public-package, post-publish, Linux, or Owner authorization proof.