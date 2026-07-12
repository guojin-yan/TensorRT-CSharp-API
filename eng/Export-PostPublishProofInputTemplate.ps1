[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { $scriptRoot = (Get-Location).Path } else { $scriptRoot = $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$template = [pscustomobject]@{
  recordKind = "post-publish-proof-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-post-publish-proof-required"
  publishedAtUtc = "<owner-fill-published-at-utc>"
  publishedManagedPackageUrl = "<owner-fill-published-managed-package-url>"
  publishedRuntimePackageUrl = "<owner-fill-published-runtime-package-url>"
  publishedManagedPackageSha256 = "<owner-fill-published-managed-package-sha256>"
  publishedRuntimePackageSha256 = "<owner-fill-published-runtime-package-sha256>"
  nugetPackageMetadataUrl = "<owner-fill-nuget-package-metadata-url>"
  githubPackagesMetadataUrl = "<owner-fill-github-packages-metadata-url>"
  downloadedManagedNupkgPath = "<owner-fill-downloaded-managed-nupkg-path>"
  downloadedManagedNupkgSha256 = "<owner-fill-downloaded-managed-nupkg-sha256>"
  downloadedRuntimeNupkgPath = "<owner-fill-downloaded-runtime-nupkg-path>"
  downloadedRuntimeNupkgSha256 = "<owner-fill-downloaded-runtime-nupkg-sha256>"
  cleanExternalConsumerRoot = "<owner-fill-clean-external-consumer-root-outside-repository>"
  consumerProjectPath = "<owner-fill-clean-external-consumer-csproj-path>"
  restoreCommand = "dotnet restore <clean-consumer-project> --source <published-package-source>"
  buildCommand = "dotnet build <clean-consumer-project> -c Release --no-restore"
  smokeCommand = "dotnet run --project <clean-consumer-project> -- --runtime-package-key $RuntimePackageKey"
  runtimePackageKey = $RuntimePackageKey
  exitCode = "<owner-fill-exit-code>"
  stdoutLogPath = "<owner-fill-stdout-log-path>"
  stdoutLogSha256 = "<owner-fill-stdout-log-sha256>"
  stderrLogPath = "<owner-fill-stderr-log-path>"
  stderrLogSha256 = "<owner-fill-stderr-log-sha256>"
  runtimeProbeReportPath = "<owner-fill-runtime-probe-report-path>"
  runtimeProbeReportSha256 = "<owner-fill-runtime-probe-report-sha256>"
  dependencyProbeStatus = "<owner-fill-passed-or-compatible-host-passed>"
  smokeStatus = "<owner-fill-passed>"
  nativeAssetsCopied = "<owner-fill-true>"
  hostOs = "<owner-fill-host-os>"
  hostArchitecture = "<owner-fill-host-architecture>"
  gpuName = "<owner-fill-gpu-name>"
  driverVersion = "<owner-fill-driver-version>"
  cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
  tensorRtVersion = "<owner-fill-tensorrt-version>"
  cudnnVersion = "<owner-fill-cudnn-version>"
  ownerName = "<owner-fill-owner-name>"
  ownerReviewedAtUtc = "<owner-fill-owner-reviewed-at-utc>"
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  postPublishProofReady = $false
  isPostPublishProof = $false
  isPackageConsumerRuntimeProof = $false
  proofBoundary = "Post-publish proof input only. It validates real public URLs, downloaded package hashes, and clean external consumer smoke evidence. Template/default records are blocked and cannot close release issues."
}

$jsonPath = Join-Path $artifactRoot "post-publish-proof-input.template.json"
$markdownPath = Join-Path $artifactRoot "post-publish-proof-input.template.md"
$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Post-Publish Proof Input Template

生成时间：$($template.generatedAtUtc)

| 字段 | 当前值 |
|---|---|
| validationState | ``$($template.validationState)`` |
| runtimePackageKey | ``$($template.runtimePackageKey)`` |
| performsPublish | ``$($template.performsPublish)`` |
| canCloseReleaseIssue | ``$($template.canCloseReleaseIssue)`` |
| postPublishProofReady | ``$($template.postPublishProofReady)`` |
| isPostPublishProof | ``$($template.isPostPublishProof)`` |

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishProofInput.ps1 -Strict
```

## Boundary

$($template.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8
Write-Host "Post-publish proof input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
