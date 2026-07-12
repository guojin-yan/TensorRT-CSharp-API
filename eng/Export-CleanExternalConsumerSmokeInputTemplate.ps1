[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$template = [pscustomobject]@{
  recordKind = "clean-external-consumer-smoke-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-clean-external-consumer-smoke-required"
  proofLineId = "clean-external-consumer-smoke"
  cleanExternalConsumerRoot = "<owner-fill-clean-external-consumer-root-outside-repository>"
  consumerProjectPath = "<owner-fill-clean-external-consumer-csproj-path>"
  restoreCommand = "dotnet restore <clean-consumer-project> --source <public-package-source>"
  buildCommand = "dotnet build <clean-consumer-project> -c Release --no-restore"
  smokeCommand = "dotnet run --project <clean-consumer-project> -- --runtime-package-key $RuntimePackageKey"
  runtimePackageKey = $RuntimePackageKey
  exitCode = "<owner-fill-smoke-exit-code>"
  startedAtUtc = "<owner-fill-started-at-utc>"
  finishedAtUtc = "<owner-fill-finished-at-utc>"
  stdoutLogPath = "<owner-fill-stdout-log-path>"
  stdoutLogSha256 = "<owner-fill-stdout-log-sha256>"
  stderrLogPath = "<owner-fill-stderr-log-path>"
  stderrLogSha256 = "<owner-fill-stderr-log-sha256>"
  runtimeProbeReportPath = "<owner-fill-runtime-probe-report-path>"
  runtimeProbeReportSha256 = "<owner-fill-runtime-probe-report-sha256>"
  hostOs = "<owner-fill-host-os>"
  hostArchitecture = "<owner-fill-host-architecture>"
  gpuName = "<owner-fill-gpu-name>"
  driverVersion = "<owner-fill-driver-version>"
  cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
  tensorRtVersion = "<owner-fill-tensorrt-version>"
  cudnnVersion = "<owner-fill-cudnn-version>"
  nativeAssetsCopied = "<owner-fill-native-assets-copied-true>"
  dependencyProbeStatus = "<owner-fill-passed-or-compatible-host-passed>"
  smokeStatus = "<owner-fill-passed>"
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canPublishGitHubPackages = $false
  canCloseReleaseIssue = $false
  canClaimRuntimeProof = $false
  canClaimPackageConsumerRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  blockedReasons = @(
    "clean-external-consumer-root-required",
    "public-package-restore-required",
    "runtime-smoke-log-hash-required",
    "runtime-probe-report-required",
    "post-publish-proof-missing"
  )
  forbiddenSubstitutes = @(
    "in-repository sample",
    "ProjectReference",
    "../src path",
    "repository absolute path",
    "RestoreSources local feed",
    "artifacts package source",
    "direct .nupkg",
    "build-only result",
    "dependency-probe-only result"
  )
  proofBoundary = "Owner input template only. Clean external consumer smoke requires an external project restored from package sources, runtime smoke exit code 0, host metadata, log hashes, runtime probe report hash, and no ProjectReference/local feed/direct nupkg substitutes. This template does not publish packages, does not close release issues, and does not claim runtime proof."
}

$jsonPath = Join-Path $artifactRoot "clean-external-consumer-smoke-input.template.json"
$markdownPath = Join-Path $artifactRoot "clean-external-consumer-smoke-input.template.md"

$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$blockedReasonRows = $template.blockedReasons | ForEach-Object { "- ``$_``" }
$forbiddenRows = $template.forbiddenSubstitutes | ForEach-Object { "- ``$_``" }

$markdown = @"
# Clean External Consumer Smoke Input Template

生成时间：$($template.generatedAtUtc)

## 用途

该模板用于 Owner 回填真实 clean external consumer restore/build/runtime smoke 证据。默认状态是 blocked，不执行发布，不使用 publish token，不关闭 release issue，也不声称已经完成 package-consumer runtime proof。

| 字段 | 当前值 |
|---|---|
| validationState | ``$($template.validationState)`` |
| cleanExternalConsumerRoot | ``$($template.cleanExternalConsumerRoot)`` |
| consumerProjectPath | ``$($template.consumerProjectPath)`` |
| runtimePackageKey | ``$($template.runtimePackageKey)`` |
| smokeCommand | ``$($template.smokeCommand)`` |
| dependencyProbeStatus | ``$($template.dependencyProbeStatus)`` |
| smokeStatus | ``$($template.smokeStatus)`` |
| performsPublish | ``$($template.performsPublish)`` |
| usesPublishToken | ``$($template.usesPublishToken)`` |
| canPublishPublicly | ``$($template.canPublishPublicly)`` |
| canClaimPackageConsumerRuntimeProof | ``$($template.canClaimPackageConsumerRuntimeProof)`` |
| isPackageConsumerRuntimeProof | ``$($template.isPackageConsumerRuntimeProof)`` |

## Blocked Reasons

$($blockedReasonRows -join "`r`n")

## Forbidden Substitutes

$($forbiddenRows -join "`r`n")

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-CleanExternalConsumerSmokeInput.ps1 -Strict
```

## Boundary

$($template.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Clean external consumer smoke input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($template.validationState) PerformsPublish=False UsesPublishToken=False CanPublishPublicly=False"
