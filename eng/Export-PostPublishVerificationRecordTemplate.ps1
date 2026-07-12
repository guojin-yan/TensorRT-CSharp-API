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

function Set-ContentWithRetry {
  param(
    [Parameter(Mandatory = $true)]
    [string]$LiteralPath,

    [Parameter(Mandatory = $true)]
    [object]$Value,

    [int]$RetryCount = 8,

    [int]$DelayMilliseconds = 125
  )

  for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
    try {
      Set-Content -LiteralPath $LiteralPath -Value $Value -Encoding utf8
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $RetryCount) {
        throw
      }

      Start-Sleep -Milliseconds ($DelayMilliseconds * $attempt)
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $RetryCount) {
        throw
      }

      Start-Sleep -Milliseconds ($DelayMilliseconds * $attempt)
    }
  }
}

$postPublishRequiredEvidence = @(
  "selectedChannel",
  "channelSourceUri",
  "publishedPackageUrl",
  "packagePageUrl",
  "managedPackageUrl",
  "runtimePackageUrl",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
  "downloadedManagedPackagePath",
  "downloadedManagedPackageSha256",
  "downloadedRuntimePackagePath",
  "downloadedRuntimePackageSha256",
  "cleanConsumerRootOutsideRepository",
  "consumerProjectPath",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "restoreLogPath",
  "nativeAssetListingSha256",
  "dependencyProbeLogPath",
  "dependencyProbeLogSha256",
  "runtimeSmokeLogPath",
  "runtimeSmokeLogSha256",
  "runtimeSmokePassed",
  "runtimeSmokeExitCode",
  "stdoutSummary",
  "stderrSummary",
  "hostMetadata",
  "runtimeNativeAssetResolutionReportPath",
  "runtimeNativeAssetResolutionReportSha256",
  "ownerVerificationDecision",
  "rollbackReviewPath",
  "rollbackReviewSha256",
  "forbiddenSubstituteScanPath",
  "forbiddenSubstituteScanSha256"
)

$verificationItems = @(
  [pscustomobject]@{ id = "published-package-version"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Published package id, version, package URL, and downloaded nupkg SHA256 from the selected channel."; boundary = "A local nupkg path is not proof that the public channel contains the package." }
  [pscustomobject]@{ id = "clean-directory"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Fresh directory path created after publication."; boundary = "The source repo workspace is not post-publish consumer proof." }
  [pscustomobject]@{ id = "clean-consumer-project-identity"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "consumerProjectName plus consumerProjectPath pointing to the clean consumer .csproj."; boundary = "A folder path alone is not enough to prove which clean consumer project was restored and smoked." }
  [pscustomobject]@{ id = "no-project-reference"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Consumer project file or restore log showing no ProjectReference."; boundary = "ProjectReference invalidates channel verification." }
  [pscustomobject]@{ id = "no-local-package-source"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Clean consumer scan showing no RestoreSources, fallback folders, NuGet.config local feed, or repository artifact feed."; boundary = "Local package sources can substitute unpublished artifacts for selected-channel packages." }
  [pscustomobject]@{ id = "no-local-nupkg-reference"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Clean consumer scan showing no direct .nupkg PackageReference path."; boundary = "Direct .nupkg references bypass selected-channel post-publish verification." }
  [pscustomobject]@{ id = "managed-package-source"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Restore log showing managed package from selected channel."; boundary = "Local bin output is not package-source proof." }
  [pscustomobject]@{ id = "runtime-package-source"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Restore log showing runtime package from selected channel or local source populated from release assets."; boundary = "GitHub Release assets are not a NuGet source until downloaded into one." }
  [pscustomobject]@{ id = "host-runtime-metadata"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Smoke host OS, GPU, driver, CUDA driver/runtime, TensorRT runtime/line, and cuDNN version."; boundary = "A smoke pass without host metadata is not reproducible release proof." }
  [pscustomobject]@{ id = "native-assets-copied"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Output directory listing for bridge and vendor runtime assets."; boundary = "Restore success alone is not native-copy proof." }
  [pscustomobject]@{ id = "dependency-probe"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "DependencyProbe BridgeInitialized output."; boundary = "Dependency probe is not runtime execution proof." }
  [pscustomobject]@{ id = "runtime-key-smoke-command"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "smokeCommand containing --runtime-package-key $RuntimePackageKey."; boundary = "A smoke command without the release runtime package key cannot prove the selected runtime package." }
  [pscustomobject]@{ id = "stdout-summary"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Reviewed stdoutSummary summarizing restore, build, dependency probe, and smoke output."; boundary = "A log path alone is not enough for release issue review." }
  [pscustomobject]@{ id = "stderr-summary"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Reviewed stderrSummary, or an explicit no-stderr-emitted note when the commands produced no stderr."; boundary = "Empty stderr must be intentionally reviewed, not omitted." }
  [pscustomobject]@{ id = "stdout-stderr-summary"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "Both stdoutSummary and stderrSummary fields filled after reviewing restore, build, dependency probe, and smoke output."; boundary = "Summaries help release review but do not replace preserved logs and matching SHA256 hashes." }
  [pscustomobject]@{ id = "compatible-host-smoke"; status = "pending"; evidence = ""; evidencePath = ""; requiredEvidence = "CUDA-compatible host smoke log, exitCode=0, smokeLogSha256, and runtimeSmokePassed=true."; boundary = "blocked-by-cuda-driver is not smoke passed." }
)

$executionSteps = @(
  [pscustomobject]@{ id = "publish-channel-confirmed"; command = "Confirm the release channel contains both managed and runtime packages."; requiredEvidence = "selectedChannel, channelSourceUri, managed/runtime package URLs."; boundary = "A planned or dry-run publish is not post-publish evidence." }
  [pscustomobject]@{ id = "download-packages"; command = "Download the managed and runtime nupkg files from the selected channel."; requiredEvidence = "managedPackageUrl, runtimePackageUrl, managedNupkgSha256, runtimeNupkgSha256."; boundary = "Local build output does not prove the channel contains the packages." }
  [pscustomobject]@{ id = "create-clean-consumer"; command = "Create a fresh consumer directory outside the source repository."; requiredEvidence = "cleanConsumerRoot, consumerProjectName, consumerProjectPath, and no ProjectReference evidence."; boundary = "The source repo workspace is not a clean post-publish consumer." }
  [pscustomobject]@{ id = "capture-host-metadata"; command = "Capture CUDA/TensorRT/cuDNN host metadata before running smoke."; requiredEvidence = "host.osDescription, host.gpuName, host.driverVersion, host.cudaDriverSupportedRuntime, host.cudaRuntimeVersion, host.tensorRtRuntimeVersion, host.tensorRtLine, host.cudnnVersion."; boundary = "Smoke evidence without host metadata cannot be reproduced or compared across runtime packages." }
  [pscustomobject]@{ id = "restore-from-channel"; command = "Restore the clean consumer from the selected channel/package source."; requiredEvidence = "restoreLogPath, restoreLogSha256, managedPackageSource, runtimePackageSource."; boundary = "An unpublished local feed cannot close post-publish verification." }
  [pscustomobject]@{ id = "verify-native-assets"; command = "Build/list output and record copied bridge/vendor native assets."; requiredEvidence = "nativeAssetsCopied=true, nativeAssetListingPath, nativeAssetListingSha256."; boundary = "Restore success alone is not native-copy proof." }
  [pscustomobject]@{ id = "run-dependency-probe"; command = "Run DependencyProbe and preserve its log."; requiredEvidence = "dependencyProbePassed=true, dependencyProbeLogPath, dependencyProbeLogSha256."; boundary = "DependencyProbe is diagnostics only, not runtime execution proof." }
  [pscustomobject]@{ id = "run-compatible-host-smoke"; command = "Run package consumer smoke on a compatible CUDA/TensorRT host with --runtime-package-key $RuntimePackageKey."; requiredEvidence = "smokeCommand with --runtime-package-key, runtimeSmokePassed=true, runtimeSmokeExitCode=0, smokeStatus=passed, smokeLogPath, smokeLogSha256, stdoutSummary, stderrSummary or no-stderr-emitted."; boundary = "blocked-by-cuda-driver is not smoke passed." }
  [pscustomobject]@{ id = "validate-record"; command = "Run Test-PostPublishVerificationRecord.ps1 -InputPath <record> -RequireExistingLog -FailOnNotProof."; requiredEvidence = "isPostPublishVerificationProof=true, canCloseReleaseIssue=true, and all referenced log SHA256 values match."; boundary = "The release issue cannot close until the validator promotes proof." }
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "post-publish-verification-record-template"
  runtimePackageKey = $RuntimePackageKey
  templateOnly = $true
  verificationState = "template-only"
  postPublishProofClassification = "template-only"
  allowedPostPublishProofClassifications = @(
    "template-only",
    "owner-action-required",
    "dependency-probe-only",
    "blocked-by-cuda-driver",
    "post-publish-package-consumer-runtime"
  )
  selectedChannel = ""
  channelSourceUri = ""
  publishedPackageUrl = ""
  packagePageUrl = ""
  downloadedManagedPackagePath = ""
  downloadedManagedPackageSha256 = ""
  downloadedRuntimePackagePath = ""
  downloadedRuntimePackageSha256 = ""
  runtimeNativeAssetResolutionReportPath = ""
  runtimeNativeAssetResolutionReportSha256 = ""
  ownerVerificationDecision = ""
  rollbackReviewPath = ""
  rollbackReviewSha256 = ""
  forbiddenSubstituteScanPath = ""
  forbiddenSubstituteScanSha256 = ""
  packageIdentity = [pscustomobject]@{
    managedPackageId = ""
    managedPackageVersion = ""
    runtimePackageId = ""
    runtimePackageVersion = ""
    managedPackageUrl = ""
    runtimePackageUrl = ""
    managedNupkgSha256 = ""
    runtimeNupkgSha256 = ""
    managedPackageSha256Source = ""
    runtimePackageSha256Source = ""
    managedPackageDownloadTimestampUtc = ""
    runtimePackageDownloadTimestampUtc = ""
  }
  cleanConsumerRoot = ""
  cleanConsumerRootOutsideRepository = $null
  consumerProjectName = ""
  consumerProjectPath = ""
  host = [pscustomobject]@{
    ownerName = ""
    machineName = ""
    osDescription = ""
    gpuName = ""
    driverVersion = ""
    cudaDriverSupportedRuntime = ""
    cudaRuntimeVersion = ""
    tensorRtRuntimeVersion = ""
    tensorRtLine = ""
    cudnnVersion = ""
  }
  restoreCommand = ""
  buildCommand = ""
  smokeCommand = ""
  stdoutSummary = ""
  stderrSummary = ""
  restoreLogPath = ""
  restoreLogSha256 = ""
  nativeAssetListingPath = ""
  nativeAssetListingSha256 = ""
  dependencyProbeLogPath = ""
  dependencyProbeLogSha256 = ""
  runtimeSmokeLogPath = ""
  runtimeSmokeLogSha256 = ""
  smokeLogPath = ""
  smokeLogSha256 = ""
  managedPackageSource = ""
  runtimePackageSource = ""
  expectedRuntimePackageKey = $RuntimePackageKey
  noProjectReference = $null
  noLocalPackageSource = $null
  noLocalNupkgPackageReference = $null
  nativeAssetsCopied = $null
  dependencyProbePassed = $null
  runtimeSmokePassed = $null
  runtimeSmokeExitCode = $null
  smokeStatus = "pending-compatible-host-execution"
  hostMetadata = [pscustomobject]@{
    ownerName = ""
    machineName = ""
    osDescription = ""
    gpuName = ""
    driverVersion = ""
    cudaDriverSupportedRuntime = ""
    cudaRuntimeVersion = ""
    tensorRtRuntimeVersion = ""
    tensorRtLine = ""
    cudnnVersion = ""
  }
  performsPublish = $false
  isPostPublishVerificationProof = $false
  canCloseReleaseIssue = $false
  ownerName = ""
  reviewerName = ""
  publishedVersion = ""
  verificationItems = $verificationItems
  executionSteps = $executionSteps
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidence.Count
  promotionRules = @(
    "This template does not publish packages.",
    "Post-publish proof requires a clean consumer after the chosen channel has been published.",
    "Package id/version, package URL, downloaded nupkg SHA256, and runtime package key must come from the selected channel.",
    "Package URL and SHA256 source fields must identify the exact selected channel download, not a local build artifact.",
    "selectedChannel, channelSourceUri, cleanConsumerRoot, package sources, log paths, and log SHA256 values must be filled from the published channel.",
    "Post-publish proof must identify the clean consumer project by consumerProjectName and consumerProjectPath.",
    "Post-publish proof must reject local package sources, repository artifact feeds, fallback folders, and direct .nupkg references.",
    "Post-publish proof must reject dashboard, dry-run, manual approval, queued workflow, missing runner, sidecar-only, and TensorRtExec report substitutes.",
    "Post-publish proof must link rollback/deprecation review and forbidden substitute scan path/hash before release close can be considered.",
    "The smoke command must include --runtime-package-key and the expected runtime package key.",
    "Host CUDA/TensorRT/cuDNN/driver/GPU/OS metadata is required for post-publish proof.",
    "stdoutSummary and stderrSummary fields are required for release issue review; use no-stderr-emitted when stderr was intentionally empty.",
    "managedPackageSource and runtimePackageSource must point to the selected channel, not local bin output or ProjectReference.",
    "Every verification item must be passed with evidence before canCloseReleaseIssue=true.",
    "Dependency probe is not runtime execution proof.",
    "Compatible host smoke is required before writing runtime smoke passed.",
    "runtimeSmokePassed=true requires runtimeSmokeExitCode=0 and a 64-character smokeLogSha256.",
    "Real post-publish validation must run with -RequireExistingLog so restore/native asset listing/dependency probe/smoke SHA256 values match the referenced files."
  )
  ownerActionSummary = @(
    "Only fill this template after owner-authorized real publication is complete.",
    "Record managed/runtime package URLs, downloaded nupkg SHA256 values, SHA256 source notes, and download timestamps from the published channel.",
    "Create the clean consumer outside the repository and restore from the real published channel, not a local feed.",
    "Confirm the clean consumer uses PackageReference and no ProjectReference before marking noProjectReference=true.",
    "Run Test-PostPublishCleanConsumerProject.ps1 and confirm noLocalPackageSource=true plus noLocalNupkgPackageReference=true before closing the issue.",
    "Preserve restore/native asset listing/dependency probe/smoke logs and matching SHA256 values.",
    "Fill stdoutSummary and stderrSummary after reviewing real output; use no-stderr-emitted only after checking stderr.",
    "Attach package page URL, downloaded package paths/hashes, runtime native asset resolution report, rollback review, and forbidden substitute scan.",
    "Run Test-PostPublishVerificationRecord.ps1 -RequireExistingLog -FailOnNotProof before setting canCloseReleaseIssue=true."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "post-publish-verification-record-template.json"
$markdownPath = Join-Path $outputRoot "post-publish-verification-record-template.md"

Set-ContentWithRetry -LiteralPath $jsonPath -Value ($record | ConvertTo-Json -Depth 8)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Post-Publish Verification Record Template")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Verification state: ``template-only``")
$lines.Add("")
$lines.Add("Post-publish verification proof: ``false``")
$lines.Add("")
$lines.Add("Post-publish proof classification: ``template-only``")
$lines.Add("")
$lines.Add("This template is filled only after a real channel publish. It does not publish packages and is not proof by itself.")
$lines.Add("")
$lines.Add("## Execution Steps")
$lines.Add("")
$lines.Add("| ID | Command | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($step in $executionSteps) {
  $lines.Add("| ``$($step.id)`` | $($step.command.Replace("|", "\|")) | $($step.requiredEvidence.Replace("|", "\|")) | $($step.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Evidence Fields")
$lines.Add("")
$lines.Add("- publishedPackageUrl / packagePageUrl")
$lines.Add("- downloadedManagedPackagePath / downloadedManagedPackageSha256")
$lines.Add("- downloadedRuntimePackagePath / downloadedRuntimePackageSha256")
$lines.Add("- selectedChannel")
$lines.Add("- channelSourceUri")
$lines.Add("- packageIdentity: managed/runtime package id, version, URL, and nupkg SHA256")
$lines.Add("- packageIdentity: managedPackageSha256Source / runtimePackageSha256Source / managedPackageDownloadTimestampUtc / runtimePackageDownloadTimestampUtc")
$lines.Add("- cleanConsumerRoot / cleanConsumerRootOutsideRepository")
$lines.Add("- consumerProjectName / consumerProjectPath")
$lines.Add("- host: ownerName, machineName, osDescription, gpuName, driverVersion, cudaDriverSupportedRuntime, cudaRuntimeVersion, tensorRtRuntimeVersion, tensorRtLine, cudnnVersion")
$lines.Add("- restoreCommand / buildCommand / smokeCommand")
$lines.Add("- stdoutSummary / stderrSummary (write no-stderr-emitted when stderr was reviewed and empty)")
$lines.Add("- restoreLogPath / restoreLogSha256")
$lines.Add("- nativeAssetListingPath / nativeAssetListingSha256")
$lines.Add("- dependencyProbeLogPath / dependencyProbeLogSha256")
$lines.Add("- smokeLogPath / smokeLogSha256")
$lines.Add("- runtimeSmokeLogPath / runtimeSmokeLogSha256")
$lines.Add("- managedPackageSource / runtimePackageSource / expectedRuntimePackageKey")
$lines.Add("- noProjectReference / noLocalPackageSource / noLocalNupkgPackageReference / nativeAssetsCopied / dependencyProbePassed / runtimeSmokePassed / runtimeSmokeExitCode")
$lines.Add("- hostMetadata")
$lines.Add("- runtimeNativeAssetResolutionReportPath / runtimeNativeAssetResolutionReportSha256")
$lines.Add("- ownerVerificationDecision")
$lines.Add("- rollbackReviewPath / rollbackReviewSha256")
$lines.Add("- forbiddenSubstituteScanPath / forbiddenSubstituteScanSha256")
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("| ID | Status | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $verificationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.status)`` | $($item.requiredEvidence.Replace("|", "\|")) | $($item.boundary.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $record.promotionRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Owner Action Summary")
$lines.Add("")
foreach ($item in $record.ownerActionSummary) {
  $lines.Add("- $item")
}

Set-ContentWithRetry -LiteralPath $markdownPath -Value $lines

Write-Host "Post-publish verification record template written to $jsonPath"
Write-Host "Post-publish verification record template written to $markdownPath"

