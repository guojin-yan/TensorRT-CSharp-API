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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $DefaultValue
}

function Normalize-PostPublishRequiredEvidence {
  param([AllowNull()][object]$Evidence)

  $items = @($Evidence | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
  foreach ($requiredField in @("noLocalPackageSource", "noLocalNupkgPackageReference")) {
    if ($items -notcontains $requiredField) {
      $items += $requiredField
    }
  }

  return @($items)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function ConvertTo-RelativePath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  $fullPath = [System.IO.Path]::GetFullPath($Path)
  $root = [System.IO.Path]::GetFullPath($RepositoryRoot)
  if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $fullPath.Substring($root.Length).TrimStart('\', '/')
  }

  return $Path
}

function New-EvidenceItem {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Artifact,
    [string]$State,
    [bool]$Passed,
    [string]$Boundary
  )

  [pscustomobject]@{
    id = $Id
    title = $Title
    artifact = $Artifact
    state = $State
    passed = $Passed
    boundary = $Boundary
  }
}

function New-PackageFileSummary {
  param([System.IO.FileInfo]$File)

  [pscustomobject]@{
    file = ConvertTo-RelativePath -Path $File.FullName
    sizeBytes = $File.Length
    sizeMb = [Math]::Round($File.Length / 1MB, 2)
  }
}

$runtimePackageMatrix = Read-JsonOrNull "artifacts\release-candidate\runtime-package-matrix.json"
$packageConsumer = Read-JsonOrNull "artifacts\package-consumer\package-consumer-validation-summary.json"
$fullRuntimeConsumer = Read-JsonOrNull "artifacts\package-consumer\runtime-smoke-full\package-consumer-validation-summary.json"
$splitCollectionConsumer = Read-JsonOrNull "artifacts\package-consumer\split-collection\package-consumer-validation-summary.json"
$bridgeConsumer = Read-JsonOrNull "artifacts\package-consumer\bridge-package-consumer-validation-summary.json"
$runtimeReadiness = Read-JsonOrNull "artifacts\package-readiness\runtime-package-readiness-summary.json"
$localFeedConsumer = Read-JsonOrNull "artifacts\local-feed-consumer\local-nuget-feed-consumer-summary.json"
$releaseCandidatePackageInventory = Read-JsonOrNull "artifacts\final-release\release-candidate-package-inventory.json"
$localRuntimeValidation = Read-JsonOrNull "artifacts\local-runtime-validation\local-runtime-validation-summary.json"
$localSplitRuntimeValidation = Read-JsonOrNull "artifacts\local-runtime-validation\local-split-runtime-validation-$RuntimePackageKey.json"
$artifactManifest = Read-JsonOrNull "artifacts\runtime\$RuntimePackageKey\artifact-manifest.json"
$runtimeManifest = Read-JsonOrNull "pack\runtime\runtime-packages.manifest.json"
$splitRuntimeManifest = Read-JsonOrNull "pack\runtime-split\split-runtime-packages.manifest.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$postPublish = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$deferredSafetyTriage = Read-JsonOrNull "artifacts\interface-coverage\deferred-candidate-safety-triage.json"

$matrixEntries = @()
if ($runtimePackageMatrix) {
  $matrixEntries = @($runtimePackageMatrix)
}

$manifestPackages = @()
if ($runtimeManifest) {
  $manifestPackages = @($runtimeManifest.packages)
}

$splitManifestPackages = @()
if ($splitRuntimeManifest) {
  $splitManifestPackages = @($splitRuntimeManifest.packages)
}

$currentMatrixEntry = $matrixEntries | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
$currentManifestPackage = $manifestPackages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
$currentSplitManifestPackages = @($splitManifestPackages | Where-Object { $_.sourceRuntimeKey -eq $RuntimePackageKey })

$runtimePackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-nupkg"
$splitPackageDirectory = Join-Path $RepositoryRoot "artifacts\runtime-split-nupkg\$RuntimePackageKey"
$managedPackageDirectory = Join-Path $RepositoryRoot "artifacts\managed"

$runtimePackageFiles = @()
if (Test-Path -LiteralPath $runtimePackageDirectory -PathType Container) {
  $currentPackageId = [string](Get-PropertyOrDefault -Object $currentManifestPackage -Name "packageId" -DefaultValue "")
  $runtimePackageFiles = @(Get-ChildItem -LiteralPath $runtimePackageDirectory -File -Filter "*.nupkg" | Where-Object {
      $_.Name.Contains($RuntimePackageKey.Replace("-", "."), [System.StringComparison]::OrdinalIgnoreCase) -or
      $_.Name.Contains($RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase) -or
      (-not [string]::IsNullOrWhiteSpace($currentPackageId) -and $_.Name.Contains($currentPackageId, [System.StringComparison]::OrdinalIgnoreCase))
    })
}

$splitPackageFiles = @()
if (Test-Path -LiteralPath $splitPackageDirectory -PathType Container) {
  $splitPackageFiles = @(Get-ChildItem -LiteralPath $splitPackageDirectory -File -Filter "*.nupkg")
}

$managedPackageFiles = @()
if (Test-Path -LiteralPath $managedPackageDirectory -PathType Container) {
  $managedPackageFiles = @(Get-ChildItem -LiteralPath $managedPackageDirectory -File -Filter "*.nupkg")
}

$nativeAssetCount = if ($artifactManifest) { @($artifactManifest.files).Count } else { -1 }
$packageConsumerNativeAssetsExpected = [int](Get-PropertyOrDefault -Object $packageConsumer -Name "NativeAssetsExpected" -DefaultValue -1)
$packageConsumerNativeAssetsFound = [int](Get-PropertyOrDefault -Object $packageConsumer -Name "NativeAssetsFound" -DefaultValue -1)
$packageConsumerMissingNativeAssets = @(Get-PropertyOrDefault -Object $packageConsumer -Name "MissingNativeAssets" -DefaultValue @())
$localFeedNativeAssetsExpected = [int](Get-PropertyOrDefault -Object $localFeedConsumer -Name "NativeAssetsExpected" -DefaultValue -1)
$localFeedNativeAssetsFound = [int](Get-PropertyOrDefault -Object $localFeedConsumer -Name "NativeAssetsFound" -DefaultValue -1)
$localFeedMissingNativeAssets = @(Get-PropertyOrDefault -Object $localFeedConsumer -Name "MissingNativeAssets" -DefaultValue @())

$packageConsumerSmokeResult = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "SmokeResult" -DefaultValue "missing-package-consumer-smoke")
$packageConsumerEvidenceKind = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "EvidenceKind" -DefaultValue "missing-package-consumer-evidence-kind")
$runtimeSmokeClassification = [string](Get-PropertyOrDefault -Object $packageConsumer -Name "RuntimeSmokeClassification" -DefaultValue "missing-runtime-smoke-classification")
$isDependencyProbeOnly = [bool](Get-PropertyOrDefault -Object $packageConsumer -Name "IsDependencyProbeOnly" -DefaultValue $true)
$isRuntimeExecutionEvidence = [bool](Get-PropertyOrDefault -Object $packageConsumer -Name "IsRuntimeExecutionEvidence" -DefaultValue $false)
$isRealCallbackRuntimeProof = [bool](Get-PropertyOrDefault -Object $packageConsumer -Name "IsRealCallbackRuntimeProof" -DefaultValue $false)
$localFeedRunStatus = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RunStatus" -DefaultValue "missing-local-feed-consumer")
$localFeedRestoreSourceMode = [string](Get-PropertyOrDefault -Object $localFeedConsumer -Name "RestoreSourceMode" -DefaultValue "missing-restore-source-mode")
$localFeedUsesProjectReference = [bool](Get-PropertyOrDefault -Object $localFeedConsumer -Name "UsesProjectReference" -DefaultValue $true)
$runtimeReadinessStatus = [string](Get-PropertyOrDefault -Object $runtimeReadiness -Name "overallStatus" -DefaultValue "missing-runtime-readiness")
$runtimeProofStatus = [string](Get-PropertyOrDefault -Object $runtimeReadiness -Name "runtimeProofStatus" -DefaultValue $packageConsumerSmokeResult)
if ([string]::IsNullOrWhiteSpace($runtimeProofStatus) -or $runtimeProofStatus -eq "missing-runtime-readiness") {
  $runtimeProofStatus = $packageConsumerSmokeResult
}

$externalRuntimeProofValidation = Read-JsonOrNull "artifacts\final-release\external-runtime-proof-validation.json"
$externalRuntimeProofState = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "validationState" -DefaultValue "missing-external-runtime-proof-validation")
$externalRuntimeProofClassification = [string](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "proofClassification" -DefaultValue "missing-proof-classification")
$externalRuntimeProofRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimePackageKeyMatches" -DefaultValue $false)
$externalRuntimeProofPackageSourceRuntimePackageKeyMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "packageSourceRuntimePackageKeyMatches" -DefaultValue $false)
$externalRuntimeProofConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "consumerProjectIdentityReady" -DefaultValue $false)
$externalRuntimeProofSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)
$externalRuntimeProofHostReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "hostReady" -DefaultValue $false)
$externalRuntimeProofCommandsReady = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "commandsReady" -DefaultValue $false)
$externalRuntimeProofPreflight = Get-PropertyOrDefault -Object $externalRuntimeProofValidation -Name "runtimeProofPreflight" -DefaultValue $null
$externalRuntimeProofPreflightAligned = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "aligned" -DefaultValue $false)
$externalRuntimeProofPreflightMatrixFound = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "matrixFound" -DefaultValue $false)
$externalRuntimeProofPreflightEntryFound = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "entryFound" -DefaultValue $false)
$externalRuntimeProofPreflightRuntimePackageIdMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "runtimePackageIdMatches" -DefaultValue $false)
$externalRuntimeProofPreflightRestoreSourceModeMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "restoreSourceModeMatches" -DefaultValue $false)
$externalRuntimeProofPreflightNativeAssetsExpectedMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "nativeAssetsExpectedMatches" -DefaultValue $false)
$externalRuntimeProofPreflightNativeAssetsFoundMatches = [bool](Get-PropertyOrDefault -Object $externalRuntimeProofPreflight -Name "nativeAssetsFoundMatches" -DefaultValue $false)
$postPublishVerificationState = [string](Get-PropertyOrDefault -Object $postPublish -Name "validationState" -DefaultValue "missing-post-publish-validation")
$postPublishProofClassification = [string](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")
$postPublishProofClassificationPromotable = [bool](Get-PropertyOrDefault -Object $postPublish -Name "postPublishProofClassificationPromotable" -DefaultValue $false)
$postPublishManagedNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $postPublish -Name "managedNupkgSha256Ready" -DefaultValue $false)
$postPublishRuntimeNupkgSha256Ready = [bool](Get-PropertyOrDefault -Object $postPublish -Name "runtimeNupkgSha256Ready" -DefaultValue $false)
$postPublishConsumerProjectIdentityReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "consumerProjectIdentityReady" -DefaultValue $false)
$postPublishSmokeCommandRuntimeKeyReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "smokeCommandRuntimeKeyReady" -DefaultValue $false)
$postPublishHostReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "hostReady" -DefaultValue $false)
$postPublishCommandsReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "commandsReady" -DefaultValue $false)
$postPublishStdoutSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutSummaryReady" -DefaultValue $false)
$postPublishStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "stderrSummaryReady" -DefaultValue $false)
$postPublishStdoutStderrSummaryReady = [bool](Get-PropertyOrDefault -Object $postPublish -Name "stdoutStderrSummaryReady" -DefaultValue $false)
$postPublishRestoreLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "restoreLogSha256Matches" -DefaultValue $false)
$postPublishNativeAssetListingSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "nativeAssetListingSha256Matches" -DefaultValue $false)
$postPublishDependencyProbeLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "dependencyProbeLogSha256Matches" -DefaultValue $false)
$postPublishSmokeLogSha256Matches = [bool](Get-PropertyOrDefault -Object $postPublish -Name "smokeLogSha256Matches" -DefaultValue $false)
$postPublishAllLogSha256Matches = $postPublishRestoreLogSha256Matches -and $postPublishNativeAssetListingSha256Matches -and $postPublishDependencyProbeLogSha256Matches -and $postPublishSmokeLogSha256Matches
$isPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $postPublish -Name "isPostPublishVerificationProof" -DefaultValue $false)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublish -Name "canCloseReleaseIssue" -DefaultValue $false)
$postPublishRequiredEvidence = @(
  Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRequiredEvidence" -DefaultValue @(
    Get-PropertyOrDefault -Object $postPublish -Name "postPublishRequiredEvidence" -DefaultValue @()
  )
)
$postPublishRequiredEvidence = Normalize-PostPublishRequiredEvidence -Evidence $postPublishRequiredEvidence
$postPublishRequiredEvidenceCount = if ($postPublishRequiredEvidence.Count -gt 0) {
  $postPublishRequiredEvidence.Count
}
else {
  [int](Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "postPublishRequiredEvidenceCount" -DefaultValue (
    [int](Get-PropertyOrDefault -Object $postPublish -Name "postPublishRequiredEvidenceCount" -DefaultValue 0)
  ))
}
$deferredSafetyTriageKind = [string](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "triageKind" -DefaultValue "missing-deferred-candidate-safety-triage")
$deferredSafetyTriageTotalRows = [int](Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "totalTriageRowCount" -DefaultValue 0)
$deferredSafetyTierSummaries = @((Get-PropertyOrDefault -Object $deferredSafetyTriage -Name "tierSummaries" -DefaultValue @()))
$deferredSafetyTierAImmediateSafeCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "A - immediate-safe" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTierBSafeAlternativeCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "B - safe-alternative-or-alias" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTierCDesignGateCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "C - design-gate-required" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTierDKeepDeferredCount = [int](Get-PropertyOrDefault -Object ($deferredSafetyTierSummaries | Where-Object { $_.safetyTier -eq "D - keep-deferred" } | Select-Object -First 1) -Name "candidateCount" -DefaultValue 0)
$deferredSafetyTriageState = if ($deferredSafetyTriageKind -eq "deferred-candidate-safety-triage") { "triage-ready-planning-input-only" } else { "missing-deferred-safety-triage" }
$deferredSafetyTriageProofBoundary = "Deferred safety triage is package proof disclosure only: safe-alternative/alias rows are planning input, design-gate rows need owner/lifetime design, keep-deferred rows remain deferred, and none can become public package proof, runtime execution proof, or permission to delete deferred records."

$nativeAssetCopyReady = $packageConsumerNativeAssetsExpected -gt 0 -and
  $packageConsumerNativeAssetsFound -ge $packageConsumerNativeAssetsExpected -and
  $packageConsumerMissingNativeAssets.Count -eq 0
$localFeedNativeAssetCopyReady = $localFeedNativeAssetsExpected -gt 0 -and
  $localFeedNativeAssetsFound -ge $localFeedNativeAssetsExpected -and
  $localFeedMissingNativeAssets.Count -eq 0
$localFeedDependencyProbeReady = [string]::Equals($localFeedRunStatus, "dependency-probe-passed", [System.StringComparison]::OrdinalIgnoreCase) -and
  [string]::Equals($localFeedRestoreSourceMode, "local-feed-only", [System.StringComparison]::OrdinalIgnoreCase) -and
  -not $localFeedUsesProjectReference
$packageInventoryState = [string](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "recordKind" -DefaultValue "missing-release-candidate-package-inventory")
$packageInventoryPackageCount = [int](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "packageCount" -DefaultValue 0)
$packageInventoryManagedPackageReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "managedPackageReady" -DefaultValue $false)
$packageInventoryFullRuntimePackageReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "fullRuntimePackageReady" -DefaultValue $false)
$packageInventoryFullRuntimePackageRequired = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "fullRuntimePackageRequired" -DefaultValue $true)
$packageInventoryRetiredPackageCandidateCount = [int](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "retiredPackageCandidateCount" -DefaultValue 0)
$packageInventorySplitBridgePackageReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "splitBridgePackageReady" -DefaultValue $false)
$packageInventorySplitRuntimePackagesReady = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "splitRuntimePackagesReady" -DefaultValue $false)
$packageInventorySha256Ready = [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "sha256Ready" -DefaultValue $false)
$packageInventoryReady = [string]::Equals($packageInventoryState, "release-candidate-package-inventory", [System.StringComparison]::OrdinalIgnoreCase) -and
  [bool](Get-PropertyOrDefault -Object $releaseCandidatePackageInventory -Name "packageSetReady" -DefaultValue $false) -and
  $packageInventoryManagedPackageReady -and
  $packageInventorySplitBridgePackageReady -and
  $packageInventorySplitRuntimePackagesReady -and
  $packageInventorySha256Ready -and
  -not $packageInventoryFullRuntimePackageRequired -and
  $packageInventoryRetiredPackageCandidateCount -eq 0

$canUseAsPublicPackageProof = $false
$canPromoteRuntimeProof = $false
$isRuntimeExecutionProof = $false
$proofState = "package-evidence-owner-review-required"

$evidenceItems = @(
  New-EvidenceItem -Id "runtime-package-matrix" -Title "Runtime package matrix" -Artifact "artifacts/release-candidate/runtime-package-matrix.json" -State ("entries=" + $matrixEntries.Count + "; currentExists=" + ($null -ne $currentMatrixEntry)) -Passed ($null -ne $currentMatrixEntry) -Boundary "Matrix presence is package targeting evidence, not publication or runtime execution proof."
  New-EvidenceItem -Id "runtime-package-manifest" -Title "Runtime package manifest" -Artifact "pack/runtime/runtime-packages.manifest.json" -State ("packages=" + $manifestPackages.Count + "; currentExists=" + ($null -ne $currentManifestPackage)) -Passed ($null -ne $currentManifestPackage) -Boundary "Manifest presence is not proof that a public package was published."
  New-EvidenceItem -Id "split-runtime-manifest" -Title "Split runtime package manifest" -Artifact "pack/runtime-split/split-runtime-packages.manifest.json" -State ("packagesForRuntime=" + $currentSplitManifestPackages.Count) -Passed ($currentSplitManifestPackages.Count -gt 0) -Boundary "Split package layout evidence still needs owner channel approval and runtime proof."
  New-EvidenceItem -Id "release-candidate-package-inventory" -Title "Release candidate package inventory" -Artifact "artifacts/final-release/release-candidate-package-inventory.json" -State ("state=" + $packageInventoryState + "; packageCount=" + $packageInventoryPackageCount + "; managedReady=" + $packageInventoryManagedPackageReady + "; bridgeReady=" + $packageInventorySplitBridgePackageReady + "; bridgeOnlySetReady=" + $packageInventorySplitRuntimePackagesReady + "; retiredCandidates=" + $packageInventoryRetiredPackageCandidateCount + "; sha256Ready=" + $packageInventorySha256Ready) -Passed $packageInventoryReady -Boundary "Local package inventory accepts only managed and bridge candidates. It is not public channel, runtime, post-publish, or Owner authorization proof."
  New-EvidenceItem -Id "runtime-nupkg-files" -Title "Runtime nupkg files" -Artifact "artifacts/runtime-nupkg; artifacts/runtime-split-nupkg; artifacts/managed" -State ("runtime=" + $runtimePackageFiles.Count + "; split=" + $splitPackageFiles.Count + "; managed=" + $managedPackageFiles.Count) -Passed (($runtimePackageFiles.Count + $splitPackageFiles.Count) -gt 0 -and $managedPackageFiles.Count -gt 0) -Boundary "Local nupkg files are not public package proof."
  New-EvidenceItem -Id "native-asset-manifest" -Title "Native asset manifest" -Artifact "artifacts/runtime/$RuntimePackageKey/artifact-manifest.json" -State ("nativeAssetCount=" + $nativeAssetCount) -Passed ($nativeAssetCount -gt 0) -Boundary "Native asset collection is layout evidence, not runtime execution proof."
  New-EvidenceItem -Id "package-consumer" -Title "Package consumer validation" -Artifact "artifacts/package-consumer/package-consumer-validation-summary.json" -State ("SmokeResult=" + $packageConsumerSmokeResult + "; EvidenceKind=" + $packageConsumerEvidenceKind + "; IsDependencyProbeOnly=" + $isDependencyProbeOnly) -Passed $nativeAssetCopyReady -Boundary "Driver-blocked or dependency-probe-only output is not runtime execution proof."
  New-EvidenceItem -Id "local-feed-consumer" -Title "Local feed consumer validation" -Artifact "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json" -State ("RunStatus=" + $localFeedRunStatus + "; RestoreSourceMode=" + $localFeedRestoreSourceMode + "; UsesProjectReference=" + $localFeedUsesProjectReference) -Passed $localFeedDependencyProbeReady -Boundary "Local feed validation is not nuget.org, GitHub Packages, or public channel proof."
  New-EvidenceItem -Id "runtime-readiness" -Title "Runtime package readiness summary" -Artifact "artifacts/package-readiness/runtime-package-readiness-summary.json" -State ("overallStatus=" + $runtimeReadinessStatus + "; runtimeProofStatus=" + $runtimeProofStatus) -Passed ($null -ne $runtimeReadiness) -Boundary "Readiness summary can be ready while runtime proof remains blocked-by-cuda-driver."
  New-EvidenceItem -Id "external-runtime-proof-record" -Title "External runtime proof validation" -Artifact "artifacts/final-release/external-runtime-proof-validation.json" -State ("proofState=" + $externalRuntimeProofState + "; classification=" + $externalRuntimeProofClassification + "; runtimeKeyMatches=" + $externalRuntimeProofRuntimePackageKeyMatches + "; packageSourceKeyMatches=" + $externalRuntimeProofPackageSourceRuntimePackageKeyMatches + "; preflightAligned=" + $externalRuntimeProofPreflightAligned + "; preflightMatrixFound=" + $externalRuntimeProofPreflightMatrixFound + "; preflightEntryFound=" + $externalRuntimeProofPreflightEntryFound + "; consumerProjectIdentityReady=" + $externalRuntimeProofConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $externalRuntimeProofSmokeCommandRuntimeKeyReady + "; hostReady=" + $externalRuntimeProofHostReady + "; commandsReady=" + $externalRuntimeProofCommandsReady) -Passed ($externalRuntimeProofPreflightAligned -and $externalRuntimeProofConsumerProjectIdentityReady -and $externalRuntimeProofSmokeCommandRuntimeKeyReady -and $externalRuntimeProofHostReady -and $externalRuntimeProofCommandsReady) -Boundary "Only package-consumer-runtime evidence with clean consumer identity, RuntimeProofPreflight alignment, --runtime-package-key smoke command, complete host metadata, and passing validator is runtime proof. RuntimeProofPreflight itself is not proof."
  New-EvidenceItem -Id "post-publish-verification" -Title "Post-publish verification" -Artifact "artifacts/final-release/post-publish-verification-validation.json" -State ("verificationState=" + $postPublishVerificationState + "; classification=" + $postPublishProofClassification + "; promotable=" + $postPublishProofClassificationPromotable + "; managedNupkgSha256Ready=" + $postPublishManagedNupkgSha256Ready + "; runtimeNupkgSha256Ready=" + $postPublishRuntimeNupkgSha256Ready + "; consumerProjectIdentityReady=" + $postPublishConsumerProjectIdentityReady + "; smokeCommandRuntimeKeyReady=" + $postPublishSmokeCommandRuntimeKeyReady + "; hostReady=" + $postPublishHostReady + "; commandsReady=" + $postPublishCommandsReady + "; stdoutSummaryReady=" + $postPublishStdoutSummaryReady + "; stderrSummaryReady=" + $postPublishStderrSummaryReady + "; stdoutStderrSummaryReady=" + $postPublishStdoutStderrSummaryReady + "; allLogSha256Matches=" + $postPublishAllLogSha256Matches + "; canCloseReleaseIssue=" + $canCloseReleaseIssue) -Passed $canCloseReleaseIssue -Boundary "Post-publish verification is channel-aftercare evidence; template-only or missing clean consumer/host/command/stdout-stderr/hash evidence is not public package proof."
  New-EvidenceItem -Id "deferred-safety-triage" -Title "Deferred safety triage" -Artifact "artifacts/interface-coverage/deferred-candidate-safety-triage.json" -State "state=$deferredSafetyTriageState; totalRows=$deferredSafetyTriageTotalRows; A=$deferredSafetyTierAImmediateSafeCount; B=$deferredSafetyTierBSafeAlternativeCount; C=$deferredSafetyTierCDesignGateCount; D=$deferredSafetyTierDKeepDeferredCount; proof=false" -Passed $false -Boundary $deferredSafetyTriageProofBoundary
  New-EvidenceItem -Id "local-runtime-validation" -Title "Local runtime validation summary" -Artifact "artifacts/local-runtime-validation/local-runtime-validation-summary.json" -State ("runtimePackageSizeMb=" + (Get-PropertyOrDefault -Object $localRuntimeValidation -Name "runtimePackageSizeMb" -DefaultValue "missing") + "; ranSmoke=" + (Get-PropertyOrDefault -Object $localRuntimeValidation -Name "ranSmoke" -DefaultValue "missing")) -Passed ($null -ne $localRuntimeValidation) -Boundary "Local package build and size evidence are not runtime execution proof."
  New-EvidenceItem -Id "local-split-runtime-validation" -Title "Local split runtime validation" -Artifact "artifacts/local-runtime-validation/local-split-runtime-validation-$RuntimePackageKey.json" -State ("splitPackageCount=" + @($localSplitRuntimeValidation).Count) -Passed ($null -ne $localSplitRuntimeValidation) -Boundary "Split package size validation is not public publication proof."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-package-proof-bundle"
  runtimePackageKey = $RuntimePackageKey
  proofState = $proofState
  canUseAsPublicPackageProof = $canUseAsPublicPackageProof
  canPromoteRuntimeProof = $canPromoteRuntimeProof
  isRuntimeExecutionProof = $isRuntimeExecutionProof
  isRuntimeExecutionEvidence = $isRuntimeExecutionEvidence
  isDependencyProbeOnly = $isDependencyProbeOnly
  isRealCallbackRuntimeProof = $isRealCallbackRuntimeProof
  runtimeProofStatus = $runtimeProofStatus
  externalRuntimeProofState = $externalRuntimeProofState
  externalRuntimeProofClassification = $externalRuntimeProofClassification
  externalRuntimeProofRuntimePackageKeyMatches = $externalRuntimeProofRuntimePackageKeyMatches
  externalRuntimeProofPackageSourceRuntimePackageKeyMatches = $externalRuntimeProofPackageSourceRuntimePackageKeyMatches
  externalRuntimeProofConsumerProjectIdentityReady = $externalRuntimeProofConsumerProjectIdentityReady
  externalRuntimeProofSmokeCommandRuntimeKeyReady = $externalRuntimeProofSmokeCommandRuntimeKeyReady
  externalRuntimeProofHostReady = $externalRuntimeProofHostReady
  externalRuntimeProofCommandsReady = $externalRuntimeProofCommandsReady
  externalRuntimeProofPreflightAligned = $externalRuntimeProofPreflightAligned
  externalRuntimeProofPreflightMatrixFound = $externalRuntimeProofPreflightMatrixFound
  externalRuntimeProofPreflightEntryFound = $externalRuntimeProofPreflightEntryFound
  externalRuntimeProofPreflightRuntimePackageIdMatches = $externalRuntimeProofPreflightRuntimePackageIdMatches
  externalRuntimeProofPreflightRestoreSourceModeMatches = $externalRuntimeProofPreflightRestoreSourceModeMatches
  externalRuntimeProofPreflightNativeAssetsExpectedMatches = $externalRuntimeProofPreflightNativeAssetsExpectedMatches
  externalRuntimeProofPreflightNativeAssetsFoundMatches = $externalRuntimeProofPreflightNativeAssetsFoundMatches
  postPublishVerificationState = $postPublishVerificationState
  postPublishProofClassification = $postPublishProofClassification
  postPublishProofClassificationPromotable = $postPublishProofClassificationPromotable
  postPublishManagedNupkgSha256Ready = $postPublishManagedNupkgSha256Ready
  postPublishRuntimeNupkgSha256Ready = $postPublishRuntimeNupkgSha256Ready
  postPublishConsumerProjectIdentityReady = $postPublishConsumerProjectIdentityReady
  postPublishSmokeCommandRuntimeKeyReady = $postPublishSmokeCommandRuntimeKeyReady
  postPublishHostReady = $postPublishHostReady
  postPublishCommandsReady = $postPublishCommandsReady
  postPublishStdoutSummaryReady = $postPublishStdoutSummaryReady
  postPublishStderrSummaryReady = $postPublishStderrSummaryReady
  postPublishStdoutStderrSummaryReady = $postPublishStdoutStderrSummaryReady
  postPublishRestoreLogSha256Matches = $postPublishRestoreLogSha256Matches
  postPublishNativeAssetListingSha256Matches = $postPublishNativeAssetListingSha256Matches
  postPublishDependencyProbeLogSha256Matches = $postPublishDependencyProbeLogSha256Matches
  postPublishSmokeLogSha256Matches = $postPublishSmokeLogSha256Matches
  postPublishAllLogSha256Matches = $postPublishAllLogSha256Matches
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidenceCount
  isPostPublishVerificationProof = $isPostPublishVerificationProof
  canCloseReleaseIssue = $canCloseReleaseIssue
  packageConsumerSmokeResult = $packageConsumerSmokeResult
  runtimeSmokeClassification = $runtimeSmokeClassification
  packageConsumerEvidenceKind = $packageConsumerEvidenceKind
  runtimeMatrixEntryCount = $matrixEntries.Count
  runtimeMatrixWindowsEntryCount = @($matrixEntries | Where-Object { $_.platform -eq "windows" }).Count
  runtimeMatrixLinuxEntryCount = @($matrixEntries | Where-Object { $_.platform -eq "linux" }).Count
  currentRuntimeMatrixEntryExists = $null -ne $currentMatrixEntry
  currentRuntimeMatrixValidationState = [string](Get-PropertyOrDefault -Object $currentMatrixEntry -Name "validationState" -DefaultValue "missing-current-matrix-entry")
  currentRuntimeMatrixRuntimeProofStatus = [string](Get-PropertyOrDefault -Object $currentMatrixEntry -Name "runtimeProofStatus" -DefaultValue "missing-current-matrix-entry")
  runtimeManifestPackageCount = $manifestPackages.Count
  currentRuntimeManifestPackageExists = $null -ne $currentManifestPackage
  currentRuntimeManifestPackageId = [string](Get-PropertyOrDefault -Object $currentManifestPackage -Name "packageId" -DefaultValue "missing-current-manifest-package")
  splitRuntimeManifestPackageCount = $splitManifestPackages.Count
  currentSplitRuntimePackageCount = $currentSplitManifestPackages.Count
  currentSplitRuntimePackageRoles = @($currentSplitManifestPackages | ForEach-Object { $_.role } | Sort-Object -Unique)
  packageInventoryState = $packageInventoryState
  packageInventoryPackageCount = $packageInventoryPackageCount
  packageInventoryManagedPackageReady = $packageInventoryManagedPackageReady
  packageInventoryFullRuntimePackageReady = $packageInventoryFullRuntimePackageReady
  packageInventoryFullRuntimePackageRequired = $packageInventoryFullRuntimePackageRequired
  packageInventoryRetiredPackageCandidateCount = $packageInventoryRetiredPackageCandidateCount
  packageInventorySplitBridgePackageReady = $packageInventorySplitBridgePackageReady
  packageInventorySplitRuntimePackagesReady = $packageInventorySplitRuntimePackagesReady
  packageInventorySha256Ready = $packageInventorySha256Ready
  packageInventoryReady = $packageInventoryReady
  runtimeNupkgFiles = @($runtimePackageFiles | ForEach-Object { New-PackageFileSummary -File $_ })
  splitNupkgFiles = @($splitPackageFiles | ForEach-Object { New-PackageFileSummary -File $_ })
  managedNupkgFiles = @($managedPackageFiles | ForEach-Object { New-PackageFileSummary -File $_ })
  nativeAssetManifestFileCount = $nativeAssetCount
  nativeAssetCopyReady = $nativeAssetCopyReady
  packageConsumerNativeAssetsExpected = $packageConsumerNativeAssetsExpected
  packageConsumerNativeAssetsFound = $packageConsumerNativeAssetsFound
  packageConsumerMissingNativeAssets = $packageConsumerMissingNativeAssets
  localFeedDependencyProbeReady = $localFeedDependencyProbeReady
  localFeedRunStatus = $localFeedRunStatus
  localFeedRestoreSourceMode = $localFeedRestoreSourceMode
  localFeedUsesProjectReference = $localFeedUsesProjectReference
  localFeedNativeAssetCopyReady = $localFeedNativeAssetCopyReady
  localFeedNativeAssetsExpected = $localFeedNativeAssetsExpected
  localFeedNativeAssetsFound = $localFeedNativeAssetsFound
  fullRuntimeConsumerSmokeResult = [string](Get-PropertyOrDefault -Object $fullRuntimeConsumer -Name "SmokeResult" -DefaultValue "missing-full-runtime-consumer")
  splitCollectionConsumerSmokeResult = [string](Get-PropertyOrDefault -Object $splitCollectionConsumer -Name "SmokeResult" -DefaultValue "missing-split-collection-consumer")
  bridgeConsumerProbeResult = [string](Get-PropertyOrDefault -Object $bridgeConsumer -Name "ProbeResult" -DefaultValue (Get-PropertyOrDefault -Object $bridgeConsumer -Name "RunStatus" -DefaultValue "missing-bridge-consumer"))
  deferredSafetyTriageState = $deferredSafetyTriageState
  deferredSafetyTriageKind = $deferredSafetyTriageKind
  deferredSafetyTriageTotalRows = $deferredSafetyTriageTotalRows
  deferredSafetyTierAImmediateSafeCount = $deferredSafetyTierAImmediateSafeCount
  deferredSafetyTierBSafeAlternativeCount = $deferredSafetyTierBSafeAlternativeCount
  deferredSafetyTierCDesignGateCount = $deferredSafetyTierCDesignGateCount
  deferredSafetyTierDKeepDeferredCount = $deferredSafetyTierDKeepDeferredCount
  deferredSafetyTriageIsPackageProof = $false
  deferredSafetyTriageIsRuntimeExecutionProof = $false
  deferredSafetyTriageCanUseAsPublicPackageProof = $false
  deferredSafetyTriageProofBoundary = $deferredSafetyTriageProofBoundary
  evidenceItems = $evidenceItems
  sourceEvidence = @(
    "artifacts/release-candidate/runtime-package-matrix.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/package-consumer/runtime-smoke-full/package-consumer-validation-summary.json",
    "artifacts/package-consumer/split-collection/package-consumer-validation-summary.json",
    "artifacts/package-consumer/bridge-package-consumer-validation-summary.json",
    "artifacts/package-readiness/runtime-package-readiness-summary.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json",
    "artifacts/local-runtime-validation/local-runtime-validation-summary.json",
    "artifacts/local-runtime-validation/local-split-runtime-validation-$RuntimePackageKey.json",
    "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/runtime/$RuntimePackageKey/artifact-manifest.json",
    "pack/runtime/runtime-packages.manifest.json",
    "pack/runtime-split/split-runtime-packages.manifest.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.md"
  )
  sourceArtifacts = @(
    "artifacts/release-candidate/runtime-package-matrix.json",
    "artifacts/package-consumer/package-consumer-validation-summary.json",
    "artifacts/package-consumer/runtime-smoke-full/package-consumer-validation-summary.json",
    "artifacts/package-consumer/split-collection/package-consumer-validation-summary.json",
    "artifacts/package-consumer/bridge-package-consumer-validation-summary.json",
    "artifacts/package-readiness/runtime-package-readiness-summary.json",
    "artifacts/local-feed-consumer/local-nuget-feed-consumer-summary.json",
    "artifacts/local-runtime-validation/local-runtime-validation-summary.json",
    "artifacts/local-runtime-validation/local-split-runtime-validation-$RuntimePackageKey.json",
    "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/runtime/$RuntimePackageKey/artifact-manifest.json",
    "pack/runtime/runtime-packages.manifest.json",
    "pack/runtime-split/split-runtime-packages.manifest.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.md"
  )
  nonSubstituteProofKinds = @(
    "template",
    "draft",
    "runbook",
    "collection package",
    "input package",
    "local feed",
    "ProjectReference",
    "build-only",
    "parse-only",
    "sidecar-only",
    "dependency-probe-only",
    "Skipped=True",
    "blocked-by-cuda-driver",
    "RuntimeProofPreflight",
    "preflight-only",
    "bridge-only package consumer log",
    "bridge-only wrapper surface",
    "WrapperSurfaceEvidenceKind=compile-surface-proof",
    "IsRuntimeExecutionProof=False",
    "mismatched log SHA256",
    "deferred safety triage",
    "safe-alternative-or-alias planning input",
    "design-gate-required planning input",
    "keep-deferred boundary disclosure"
  )
  safetyNotes = @(
    "This bundle does not publish packages.",
    "canUseAsPublicPackageProof=false until a real owner-approved public/private channel package proof is attached.",
    "isRuntimeExecutionProof=false while runtimeProofStatus is blocked-by-cuda-driver.",
    "DependencyProbe output and native asset copy evidence are package diagnostics, not runtime execution proof.",
    "External runtime proof stays non-promotable until clean consumer project identity, --runtime-package-key smoke command, and complete CUDA/TensorRT/cuDNN host metadata pass validation.",
    "RuntimeProofPreflight alignment is a strict validator prerequisite, but RuntimeProofPreflight itself is owner-action-required audit metadata and never promotes runtime proof.",
    "Post-publish verification stays non-closeable until clean consumer identity, host metadata, command capture, --runtime-package-key smoke command, stdout/stderr summaries, and managed/runtime nupkg hashes pass validation.",
    "Local feed consumer success is not nuget.org or GitHub Packages publication proof.",
    "NVIDIA runtime redistribution approval remains an owner/legal decision outside this generated bundle.",
    $deferredSafetyTriageProofBoundary
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-package-proof-bundle.json"
$markdownPath = Join-Path $outputRoot "release-package-proof-bundle.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Package Proof Bundle")
$lines.Add("")
$lines.Add("Runtime key: ``$RuntimePackageKey``")
$lines.Add("")
$lines.Add("Proof state: ``$proofState``")
$lines.Add("")
$lines.Add("This bundle aggregates package layout, local feed, split package, native asset copy, and package-consumer evidence. It does not publish packages and it is not runtime execution proof.")
$lines.Add("")
$lines.Add("## Decision Fields")
$lines.Add("")
$lines.Add("- can use as public package proof: ``$canUseAsPublicPackageProof``")
$lines.Add("- can promote runtime proof: ``$canPromoteRuntimeProof``")
$lines.Add("- runtime execution proof: ``$isRuntimeExecutionProof``")
$lines.Add("- runtime proof status: ``$runtimeProofStatus``")
$lines.Add("- external runtime proof state: ``$externalRuntimeProofState``")
$lines.Add("- external runtime proof consumer project identity ready: ``$externalRuntimeProofConsumerProjectIdentityReady``")
$lines.Add("- external runtime proof smoke command runtime key ready: ``$externalRuntimeProofSmokeCommandRuntimeKeyReady``")
$lines.Add("- external runtime proof host ready: ``$externalRuntimeProofHostReady``")
$lines.Add("- external runtime proof commands ready: ``$externalRuntimeProofCommandsReady``")
$lines.Add("- external runtime proof preflight aligned: ``$externalRuntimeProofPreflightAligned``")
$lines.Add("- external runtime proof preflight matrix found: ``$externalRuntimeProofPreflightMatrixFound``")
$lines.Add("- external runtime proof preflight entry found: ``$externalRuntimeProofPreflightEntryFound``")
$lines.Add("- external runtime proof preflight runtime package id matches: ``$externalRuntimeProofPreflightRuntimePackageIdMatches``")
$lines.Add("- external runtime proof preflight restore source mode matches: ``$externalRuntimeProofPreflightRestoreSourceModeMatches``")
$lines.Add("- external runtime proof preflight native assets expected matches: ``$externalRuntimeProofPreflightNativeAssetsExpectedMatches``")
$lines.Add("- external runtime proof preflight native assets found matches: ``$externalRuntimeProofPreflightNativeAssetsFoundMatches``")
$lines.Add("- post-publish verification state: ``$postPublishVerificationState``")
$lines.Add("- post-publish proof classification: ``$postPublishProofClassification``")
$lines.Add("- post-publish managed nupkg SHA256 ready: ``$postPublishManagedNupkgSha256Ready``")
$lines.Add("- post-publish runtime nupkg SHA256 ready: ``$postPublishRuntimeNupkgSha256Ready``")
$lines.Add("- post-publish consumer project identity ready: ``$postPublishConsumerProjectIdentityReady``")
$lines.Add("- post-publish smoke command runtime key ready: ``$postPublishSmokeCommandRuntimeKeyReady``")
$lines.Add("- post-publish host ready: ``$postPublishHostReady``")
$lines.Add("- post-publish commands ready: ``$postPublishCommandsReady``")
$lines.Add("- post-publish stdout summary ready: ``$postPublishStdoutSummaryReady``")
$lines.Add("- post-publish stderr summary ready: ``$postPublishStderrSummaryReady``")
$lines.Add("- post-publish stdout/stderr summary ready: ``$postPublishStdoutStderrSummaryReady``")
$lines.Add("- post-publish all log SHA256 matches: ``$postPublishAllLogSha256Matches``")
$lines.Add("- post-publish required evidence count: ``$postPublishRequiredEvidenceCount``")
$lines.Add("- can close release issue: ``$canCloseReleaseIssue``")
$lines.Add("- dependency probe only: ``$isDependencyProbeOnly``")
$lines.Add("- real callback runtime proof: ``$isRealCallbackRuntimeProof``")
$lines.Add("- deferred safety triage: ``$deferredSafetyTriageState``")
$lines.Add("- deferred safety triage rows: ``$deferredSafetyTriageTotalRows``")
$lines.Add("- deferred safety A immediate-safe: ``$deferredSafetyTierAImmediateSafeCount``")
$lines.Add("- deferred safety B safe-alternative-or-alias: ``$deferredSafetyTierBSafeAlternativeCount``")
$lines.Add("- deferred safety C design-gate-required: ``$deferredSafetyTierCDesignGateCount``")
$lines.Add("- deferred safety D keep-deferred: ``$deferredSafetyTierDKeepDeferredCount``")
$lines.Add("- deferred safety triage is package proof: ``False``")
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Package Snapshot")
$lines.Add("")
$lines.Add("- runtime matrix entries: $($matrixEntries.Count)")
$lines.Add("- runtime manifest packages: $($manifestPackages.Count)")
$lines.Add("- split manifest packages for current runtime: $($currentSplitManifestPackages.Count)")
$lines.Add("- package inventory state: ``$packageInventoryState``")
$lines.Add("- package inventory package count: ``$packageInventoryPackageCount``")
$lines.Add("- package inventory split bridge ready: ``$packageInventorySplitBridgePackageReady``")
$lines.Add("- package inventory ready: ``$packageInventoryReady``")
$lines.Add("- package inventory SHA256 ready: ``$packageInventorySha256Ready``")
$lines.Add("- runtime nupkg files: $($runtimePackageFiles.Count)")
$lines.Add("- split nupkg files: $($splitPackageFiles.Count)")
$lines.Add("- managed nupkg files: $($managedPackageFiles.Count)")
$lines.Add("- native asset manifest file count: $nativeAssetCount")
$lines.Add("- package consumer native assets: $packageConsumerNativeAssetsFound/$packageConsumerNativeAssetsExpected")
$lines.Add("- local feed native assets: $localFeedNativeAssetsFound/$localFeedNativeAssetsExpected")
$lines.Add("")
$lines.Add("## Evidence Items")
$lines.Add("")
$lines.Add("| ID | State | Passed | Artifact | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($item in $evidenceItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.state)`` | ``$($item.passed)`` | ``$($item.artifact)`` | $(ConvertTo-MarkdownCell $item.boundary) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $record.safetyNotes) {
  $lines.Add("- $note")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release package proof bundle written to $jsonPath"
Write-Host "Release package proof bundle written to $markdownPath"
