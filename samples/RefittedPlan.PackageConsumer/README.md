# Refitted Plan Local Package Consumer

This sample is copied into an isolated repository-external workspace by
`eng/Test-TrtexecRefittedPlanPackageConsumer.ps1`. The generated project contains only two
`PackageReference` entries: the managed API package and the selected TRT10 bridge-only package.
It has no `ProjectReference` and does not load managed assemblies from the source tree.

The consumer receives a copied full-weight refitted plan and copied float input. It deserializes
the plan through the public `TensorRtRuntime` wrapper, creates an independently owned engine,
execution context, binding set, and CUDA stream, then performs one enqueue and raw output readback.
The process succeeds only when the raw float output SHA256 exactly matches the committed
same-process, second-process, and full-weight baseline hash.

Run the complete restore/build/runtime/evidence flow from the repository root:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass `
  -File .\eng\Test-TrtexecRefittedPlanPackageConsumer.ps1 -Strict
```

This is local-file-feed package-consumer runtime evidence on one compatible host. It is not proof
that packages were downloaded from a public feed, not post-publish verification, and not public
release or issue-close authorization.
