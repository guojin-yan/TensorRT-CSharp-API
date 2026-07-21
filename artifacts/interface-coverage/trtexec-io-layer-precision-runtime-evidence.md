# TensorRtExec I/O And Layer Precision Runtime Evidence

- Evidence kind: `local-project-reference-synthetic-build-policy-smoke`
- Source state: working tree after `f4c38e74680c02cdb4c1be06ecc8d7347a591159`
- Host: RTX 3060 Laptop / driver 576.02

## TRT10 Policy Run

TRT10.11/CUDA12.9 completed a real embedded identity build with `fp32:chw` input/output policy, `prefer` precision constraints, `*:fp32` layer precision, and `*:fp32` layer output type. All five policies reported `Applied=True` and `ReadbackMatch=True`, entered `AppliedOptions`, and did not enter `ParseOnlyOptions`.

The run completed engine file round-trip, two measured enqueues, and synthetic identity output match. Report SHA256 is `5AF93D993E5B6A3B0459D495D6607AC948BEF0EB0B126687CCA4FD9A0FE9998A`; engine SHA256 is `64EC4A51BAF2B22918EED397C3781BD2680D4E466508A92441EBE884E641E23A`.

## TRT8 Builder Flag Mapping

TRT8.6/CUDA12.1 `NetworkBuilderSmokeRunner` set DirectIO, verified that PreferPrecisionConstraints remained false, then set PreferPrecisionConstraints and cleared DirectIO while Prefer remained true. The permanent smoke output is `BuilderFlagMapping=TRT8DirectIORaw12PreferRaw11:Isolated:True`. The same process built and deserialized an engine, enqueued it, and matched output.

This closes the prior mapping defect where the stable public TRT10-style logical value `DirectIO=11` could be sent directly to TRT8 and accidentally select vendor raw 11 (`PreferPrecisionConstraints`) instead of raw 12 (`DirectIO`). Public enum values remain unchanged; managed interop now maps each API line.

## TRT11 Guard

TRT11/CUDA12.9 hit the known vendor structured exception `3228369022` during runtime creation before network creation. The report therefore remains `dependency-probe-only`; all five policy options remain in `ParseOnlyOptions`. No removed TensorRT 11 precision setter is reported as applied.

## Boundary

This evidence proves local typed routing/readback and synthetic identity execution. It is not external-model caller-layout proof, tactic selection proof, numerical accuracy proof, DLA execution proof, repository-external package-consumer proof, publish approval, or release-close approval. `canPublishPublicly=false`. No NuGet push, GitHub Packages publish, GitHub Release upload, or issue close was performed.
