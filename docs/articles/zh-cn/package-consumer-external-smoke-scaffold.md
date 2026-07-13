# Package Consumer External Smoke Scaffold

`package-consumer-external-smoke-scaffold` 用于生成仓库外 clean consumer 项目骨架，帮助 Owner 在真实公开包源可用后执行 package-consumer-runtime smoke。它只提供项目形态、命令形状和扫描摘要，不是运行 proof。

## 产物

- `artifacts/final-release/package-consumer-external-smoke-scaffold.json`
- `artifacts/final-release/package-consumer-external-smoke-scaffold.md`
- 默认仓库外项目目录：`%TEMP%\TensorRtSharp4.PackageConsumerSmoke.Template`

## 生成命令

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\New-PackageConsumerExternalSmokeScaffold.ps1
```

## Proof 边界

- `outputRootIsOutsideRepository=true` 只是 clean consumer 形态的前置条件，不是 smoke proof。
- `publicPackageSourceIsLocal=true` 时仍是 placeholder/local source，不可作为 public package proof。
- 生成的 `.csproj` 不应包含 `ProjectReference`，也不应直接引用 `.nupkg`。
- `isProof=false`、`performsPublish=false`、`canPublishPublicly=false`、`canCloseReleaseIssue=false` 必须保持。
- 真实晋级仍需要 Owner 在兼容 CUDA/TensorRT 主机上使用公开包源 restore/build/run，并把 package SHA256、host metadata、smoke log SHA256 和 validator 结果回填到 `package-consumer-runtime-proof-record`。

## 与 Release Evidence Bundle 的关系

`release-evidence-bundle` 会展示 scaffold 的外部目录、包源、ProjectReference/direct `.nupkg` 扫描结果和 `canBePublicProof`，用于确认这些 forbidden substitute 没有进入 proof 路径。即使扫描通过，scaffold 也只是执行起点；只有 validator-passing real clean external consumer smoke record 才能提升 `package-consumer-runtime` proof。
