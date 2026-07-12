[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$template = [pscustomobject]@{
  recordKind = "public-package-proof-owner-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  templateState = "blocked-public-package-proof-owner-input-required"
  runtimePackageKey = $RuntimePackageKey
  nugetPackageSource = "<owner-fill-nuget-package-source-url-or-id>"
  githubRelease = [pscustomobject]@{
    releaseUrl = "<owner-fill-github-release-url>"
    tagName = "<owner-fill-github-release-tag>"
    managedAssetPath = "<owner-fill-managed-github-release-asset-path-or-url>"
    managedAssetSha256 = "<owner-fill-managed-github-release-asset-sha256>"
    runtimeAssetPath = "<owner-fill-runtime-github-release-asset-path-or-url>"
    runtimeAssetSha256 = "<owner-fill-runtime-github-release-asset-sha256>"
  }
  managedPackage = [pscustomobject]@{
    packageId = "JYPPX.TensorRT.CSharp.API"
    version = "<owner-fill-managed-package-version>"
    publicSourceUrl = "<owner-fill-public-package-source-url>"
    registryUrl = "<owner-fill-registry-url>"
    packageUrl = "<owner-fill-managed-package-url>"
    nupkgPath = "<owner-fill-managed-nupkg-path>"
    nupkgSha256 = "<owner-fill-managed-nupkg-sha256>"
    sha256Source = "<owner-fill-managed-sha256-source>"
    publicDownloadUrl = "<owner-fill-managed-public-download-url>"
    publicDownloadSha256 = "<owner-fill-managed-public-download-sha256>"
  }
  runtimePackage = [pscustomobject]@{
    packageId = "JYPPX.TensorRT.CSharp.API.runtime.win-x64-trt11.0-cuda13.2-cudnn9.22"
    version = "<owner-fill-runtime-package-version>"
    publicSourceUrl = "<owner-fill-public-package-source-url>"
    registryUrl = "<owner-fill-registry-url>"
    packageUrl = "<owner-fill-runtime-package-url>"
    nupkgPath = "<owner-fill-runtime-nupkg-path>"
    nupkgSha256 = "<owner-fill-runtime-nupkg-sha256>"
    sha256Source = "<owner-fill-runtime-sha256-source>"
    publicDownloadUrl = "<owner-fill-runtime-public-download-url>"
    publicDownloadSha256 = "<owner-fill-runtime-public-download-sha256>"
  }
  cleanExternalConsumer = [pscustomobject]@{
    root = "<owner-fill-clean-external-consumer-root-outside-repository>"
    projectPath = "<owner-fill-clean-external-consumer-csproj-path>"
    restoreCommand = "dotnet restore <clean-consumer-project> --source <public-package-source>"
    buildCommand = "dotnet build <clean-consumer-project> -c Release --no-restore"
    smokeCommand = "dotnet run --project <clean-consumer-project> -c Release -- --runtime-package-key $RuntimePackageKey"
    restoreLogPath = "<owner-fill-clean-consumer-restore-log-path>"
    restoreLogSha256 = "<owner-fill-clean-consumer-restore-log-sha256>"
    buildLogPath = "<owner-fill-clean-consumer-build-log-path>"
    buildLogSha256 = "<owner-fill-clean-consumer-build-log-sha256>"
    smokeLogPath = "<owner-fill-clean-consumer-smoke-log-path>"
    smokeLogSha256 = "<owner-fill-clean-consumer-smoke-log-sha256>"
    stdoutLogPath = "<owner-fill-clean-consumer-stdout-log-path>"
    stdoutLogSha256 = "<owner-fill-clean-consumer-stdout-log-sha256>"
    stderrLogPath = "<owner-fill-clean-consumer-stderr-log-path-or-empty-if-no-stderr>"
    stderrLogSha256 = "<owner-fill-clean-consumer-stderr-log-sha256-or-empty-if-no-stderr>"
    exitCode = "<owner-fill-clean-consumer-smoke-exit-code>"
  }
  hostMetadata = [pscustomobject]@{
    ownerName = "<owner-fill-owner-name>"
    machineName = "<owner-fill-machine-name>"
    osDescription = "<owner-fill-os-description>"
    architecture = "<owner-fill-architecture>"
    gpuName = "<owner-fill-gpu-name>"
    cudaDriverVersion = "<owner-fill-cuda-driver-version>"
    cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
    cudnnVersion = "<owner-fill-cudnn-version>"
    tensorRtVersion = "<owner-fill-tensorrt-version>"
    tensorRtLine = "<owner-fill-tensorrt-line>"
  }
  ownerReview = [pscustomobject]@{
    reviewer = "<owner-fill-owner-reviewer>"
    reviewedAtUtc = "<owner-fill-owner-review-timestamp-utc>"
    approvalState = "owner-action-required"
    notes = "<owner-fill-owner-review-notes>"
  }
  publishTimestampUtc = "<owner-fill-public-publish-timestamp-utc>"
  ownerReviewer = "<owner-fill-owner-reviewer>"
  ownerReviewTimestampUtc = "<owner-fill-owner-review-timestamp-utc>"
  ownerConfirmation = [pscustomobject]@{
    confirmsPublicPackageSource = $false
    confirmsNoLocalFeed = $false
    confirmsNoProjectReference = $false
    confirmsNoDirectNupkgReference = $false
    confirmsPackageHashesReviewed = $false
    confirmsPublicDownload = $false
    confirmsGithubReleaseAssetsReviewed = $false
    confirmsCleanExternalConsumerRestoreBuildSmoke = $false
    confirmsStdoutStderrSha256Reviewed = $false
    confirmsHostMetadataReviewed = $false
  }
  requiredRealInputFields = @(
    "nugetPackageSource",
    "githubRelease.releaseUrl",
    "githubRelease.tagName",
    "githubRelease.managedAssetPath",
    "githubRelease.managedAssetSha256",
    "githubRelease.runtimeAssetPath",
    "githubRelease.runtimeAssetSha256",
    "managedPackage.version",
    "managedPackage.publicSourceUrl",
    "managedPackage.registryUrl",
    "managedPackage.packageUrl",
    "managedPackage.nupkgPath",
    "managedPackage.nupkgSha256",
    "managedPackage.publicDownloadUrl",
    "managedPackage.publicDownloadSha256",
    "runtimePackage.version",
    "runtimePackage.publicSourceUrl",
    "runtimePackage.registryUrl",
    "runtimePackage.packageUrl",
    "runtimePackage.nupkgPath",
    "runtimePackage.nupkgSha256",
    "runtimePackage.publicDownloadUrl",
    "runtimePackage.publicDownloadSha256",
    "cleanExternalConsumer.root",
    "cleanExternalConsumer.projectPath",
    "cleanExternalConsumer.restoreCommand",
    "cleanExternalConsumer.buildCommand",
    "cleanExternalConsumer.smokeCommand",
    "cleanExternalConsumer.restoreLogPath",
    "cleanExternalConsumer.restoreLogSha256",
    "cleanExternalConsumer.buildLogPath",
    "cleanExternalConsumer.buildLogSha256",
    "cleanExternalConsumer.smokeLogPath",
    "cleanExternalConsumer.smokeLogSha256",
    "cleanExternalConsumer.stdoutLogPath",
    "cleanExternalConsumer.stdoutLogSha256",
    "cleanExternalConsumer.stderrLogPath",
    "cleanExternalConsumer.stderrLogSha256",
    "cleanExternalConsumer.exitCode",
    "hostMetadata.ownerName",
    "hostMetadata.machineName",
    "hostMetadata.osDescription",
    "hostMetadata.architecture",
    "hostMetadata.gpuName",
    "hostMetadata.cudaDriverVersion",
    "hostMetadata.cudaRuntimeVersion",
    "hostMetadata.cudnnVersion",
    "hostMetadata.tensorRtVersion",
    "hostMetadata.tensorRtLine",
    "ownerReview.reviewer",
    "ownerReview.reviewedAtUtc",
    "ownerReview.approvalState",
    "publishTimestampUtc",
    "ownerReviewer",
    "ownerReviewTimestampUtc",
    "ownerConfirmation.confirmsPublicPackageSource",
    "ownerConfirmation.confirmsNoLocalFeed",
    "ownerConfirmation.confirmsNoProjectReference",
    "ownerConfirmation.confirmsNoDirectNupkgReference",
    "ownerConfirmation.confirmsPackageHashesReviewed",
    "ownerConfirmation.confirmsPublicDownload",
    "ownerConfirmation.confirmsGithubReleaseAssetsReviewed",
    "ownerConfirmation.confirmsCleanExternalConsumerRestoreBuildSmoke",
    "ownerConfirmation.confirmsStdoutStderrSha256Reviewed",
    "ownerConfirmation.confirmsHostMetadataReviewed"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  safetyBoundary = "Owner public package proof input template only. It is not runtime proof, post-publish proof, publish approval, release close approval, or package push."
}

$jsonPath = Join-Path $artifactRoot "public-package-proof-owner-input.template.json"
$markdownPath = Join-Path $artifactRoot "public-package-proof-owner-input.template.md"
$template | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Public Package Proof Owner Input Template",
  "",
  "生成时间：$($template.generatedAtUtc)",
  "",
  "该模板用于 Owner 在真实公开包发布后回填 managed/runtime .nupkg 的公开源、NuGet package source、GitHub Release asset、包 URL、SHA256、clean external consumer restore/build/smoke 日志、stdout/stderr SHA256、host metadata、发布时间和人工复核字段。",
  "",
  "## Boundary",
  "",
  "- ``performsPublish=false``",
  "- ``canPromoteRuntimeProof=false``",
  "- ``canPublishPublicly=false``",
  "- ``canCloseReleaseIssue=false``",
  "- ``isRuntimeExecutionProof=false``",
  "- ``isReleaseCloseProof=false``",
  "- ``isPostPublishProof=false``",
  "",
  "该模板不会执行发布，也不能替代真实 public package proof、post-publish proof、runtime proof 或 release close approval。",
  "",
  "## Required Owner Evidence",
  "",
  "- NuGet package source、managed/runtime package URL、public download URL 与 SHA256。",
  "- GitHub Release URL、tag、managed/runtime release asset path/hash。",
  "- 仓库外 clean external consumer 的 restore/build/smoke command、日志路径和 SHA256。",
  "- stdout/stderr 日志路径和 SHA256，stderr 为空时也要显式记录 Owner 复核结论。",
  "- hostMetadata 与 ownerReview，包含 reviewer、review timestamp 和 approvalState。",
  "",
  "## Validator",
  "",
  "``pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageProofOwnerInput.ps1 -Strict``"
) -join [Environment]::NewLine

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public package proof owner input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "TemplateState=$($template.templateState) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"
