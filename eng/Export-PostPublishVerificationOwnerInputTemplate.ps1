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

function Get-RelativeFileSha256OrPlaceholder {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return "<owner-fill-$($RelativePath.Replace('\','-').Replace('/','-'))-sha256>"
  }

  return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$cleanConsumerScanPath = "artifacts/final-release/post-publish-clean-consumer-project-scan.json"
$template = [pscustomobject]@{
  recordKind = "post-publish-verification-owner-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ownerInputState = "template-owner-input-required"
  proofLineId = "post-publish-verification"
  runtimePackageKey = $RuntimePackageKey
  selectedChannel = "<owner-fill-selected-public-channel>"
  channelSourceUri = "<owner-fill-public-channel-source-uri>"
  publishedPackageUrl = "<owner-fill-published-package-url>"
  packagePageUrl = "<owner-fill-package-page-url>"
  downloadedManagedPackagePath = "<owner-fill-downloaded-managed-package-path>"
  downloadedManagedPackageSha256 = "<owner-fill-downloaded-managed-package-sha256>"
  downloadedRuntimePackagePath = "<owner-fill-downloaded-runtime-package-path>"
  downloadedRuntimePackageSha256 = "<owner-fill-downloaded-runtime-package-sha256>"
  runtimeNativeAssetResolutionReportPath = "<owner-fill-runtime-native-asset-resolution-report-path>"
  runtimeNativeAssetResolutionReportSha256 = "<owner-fill-runtime-native-asset-resolution-report-sha256>"
  ownerVerificationDecision = "<owner-fill-owner-verification-decision>"
  rollbackReviewPath = "<owner-fill-rollback-review-path>"
  rollbackReviewSha256 = "<owner-fill-rollback-review-sha256>"
  forbiddenSubstituteScanPath = "<owner-fill-forbidden-substitute-scan-path>"
  forbiddenSubstituteScanSha256 = "<owner-fill-forbidden-substitute-scan-sha256>"
  cleanConsumerProjectScanPath = $cleanConsumerScanPath
  cleanConsumerProjectScanSha256 = Get-RelativeFileSha256OrPlaceholder -RelativePath $cleanConsumerScanPath
  cleanConsumerRoot = "<owner-fill-clean-consumer-root-outside-repository>"
  consumerProjectName = "<owner-fill-consumer-project-name>"
  consumerProjectPath = "<owner-fill-clean-consumer-csproj-path>"
  packageIdentity = [pscustomobject]@{
    managedPackageId = "JYPPX.TensorRT.CSharp.API"
    managedPackageVersion = "<owner-fill-managed-package-version>"
    runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22"
    runtimePackageVersion = "<owner-fill-runtime-package-version>"
    managedPackageUrl = "<owner-fill-managed-package-download-url>"
    runtimePackageUrl = "<owner-fill-runtime-package-download-url>"
    managedNupkgSha256 = "<owner-fill-managed-nupkg-sha256>"
    runtimeNupkgSha256 = "<owner-fill-runtime-nupkg-sha256>"
    managedPackageSha256Source = "<owner-fill-managed-sha256-source>"
    runtimePackageSha256Source = "<owner-fill-runtime-sha256-source>"
    managedPackageDownloadTimestampUtc = "<owner-fill-managed-download-timestamp-utc>"
    runtimePackageDownloadTimestampUtc = "<owner-fill-runtime-download-timestamp-utc>"
  }
  host = [pscustomobject]@{
    ownerName = "<owner-fill-owner-name>"
    machineName = "<owner-fill-machine-name>"
    osDescription = "<owner-fill-os-description>"
    gpuName = "<owner-fill-gpu-name>"
    driverVersion = "<owner-fill-driver-version>"
    cudaDriverSupportedRuntime = "<owner-fill-cuda-driver-supported-runtime>"
    cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
    tensorRtRuntimeVersion = "<owner-fill-tensorrt-runtime-version>"
    tensorRtLine = "<owner-fill-tensorrt-line>"
    cudnnVersion = "<owner-fill-cudnn-version>"
  }
  restoreCommand = "<owner-fill-restore-command>"
  buildCommand = "<owner-fill-build-command>"
  smokeCommand = "dotnet run --project <owner-fill-clean-consumer-csproj-path> -c Release -- --runtime-package-key $RuntimePackageKey"
  stdoutSummary = "<owner-fill-reviewed-stdout-summary>"
  stderrSummary = "<owner-fill-reviewed-stderr-summary-or-no-stderr-emitted>"
  restoreLogPath = "<owner-fill-restore-log-path>"
  restoreLogSha256 = "<owner-fill-restore-log-sha256>"
  nativeAssetListingPath = "<owner-fill-native-asset-listing-path>"
  nativeAssetListingSha256 = "<owner-fill-native-asset-listing-sha256>"
  dependencyProbeLogPath = "<owner-fill-dependency-probe-log-path>"
  dependencyProbeLogSha256 = "<owner-fill-dependency-probe-log-sha256>"
  smokeLogPath = "<owner-fill-smoke-log-path>"
  smokeLogSha256 = "<owner-fill-smoke-log-sha256>"
  managedPackageSource = "<owner-fill-managed-package-source-from-public-channel>"
  runtimePackageSource = "<owner-fill-runtime-package-source-from-public-channel>"
  expectedRuntimePackageKey = $RuntimePackageKey
  noProjectReference = $false
  noLocalPackageSource = $false
  noLocalNupkgPackageReference = $false
  nativeAssetsCopied = $false
  dependencyProbePassed = $false
  runtimeSmokePassed = $false
  runtimeSmokeExitCode = $null
  smokeStatus = "owner-action-required"
  ownerName = "<owner-fill-owner-name>"
  reviewerName = "<owner-fill-reviewer-name>"
  publishedVersion = "<owner-fill-published-version>"
  strictValidationCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath artifacts/final-release/post-publish-verification-record.json -RequireExistingLog -FailOnNotProof"
  performsPublish = $false
  canPublishPublicly = $false
  isPostPublishVerificationProof = $false
  canCloseReleaseIssue = $false
  forbiddenNonProofSubstitutes = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "dashboard",
    "dry-run",
    "manual approval",
    "queued GitHub Actions run",
    "missing self-hosted runner",
    "sidecar-only",
    "TensorRtExec report"
  )
  safetyBoundary = "Owner input template only. It does not publish packages, prove post-publish verification, or close the release issue. Local feed, ProjectReference, direct .nupkg, dashboard, dry-run, manual approval, queued workflow, missing runner, sidecar-only, and TensorRtExec report are non-proof substitutes."
}

$jsonPath = Join-Path $artifactRoot "post-publish-verification-owner-input.template.json"
$markdownPath = Join-Path $artifactRoot "post-publish-verification-owner-input.template.md"

$template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Post-Publish Verification Owner Input Template

生成时间：$($template.generatedAtUtc)

## 用途

该模板用于 Owner 在真实发布后回填 post-publish verification 所需字段：公开包源、下载包 SHA256、仓库外 clean consumer、restore/build/smoke 日志、stdout/stderr 摘要和兼容主机元数据。

它不是 proof，不执行发布，不关闭 release issue。

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationOwnerInput.ps1 -Strict
```

## Record Projection

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishVerificationRecordFromOwnerInput.ps1 -OwnerInputPath artifacts/final-release/post-publish-verification-owner-input.template.json
```

## Boundary

- ``performsPublish=false``
- ``canPublishPublicly=false``
- ``isPostPublishVerificationProof=false``
- ``canCloseReleaseIssue=false``
- template、owner input、local feed、ProjectReference、direct .nupkg、dashboard、dry-run、manual approval、queued workflow、missing runner、sidecar-only、TensorRtExec report、schema-only 和 preflight-only 都不是 post-publish proof。
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification owner input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "OwnerInputState=$($template.ownerInputState) PerformsPublish=$($template.performsPublish) CanCloseReleaseIssue=$($template.canCloseReleaseIssue)"
