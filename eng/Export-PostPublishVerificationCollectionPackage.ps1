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

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-CollectionStep {
  param([string]$Id, [int]$Order, [string]$Command, [string]$RequiredEvidence, [string]$Boundary)
  [pscustomobject]@{
    id = $Id
    order = $Order
    command = $Command
    requiredEvidence = $RequiredEvidence
    boundary = $Boundary
    performsPublish = $false
    isPostPublishVerificationProof = $false
    canCloseReleaseIssue = $false
  }
}

$backfillPlan = Read-JsonOrNull "artifacts\final-release\post-publish-verification-backfill-plan.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-verification-validation.json"
$ownerPlan = Read-JsonOrNull "artifacts\final-release\owner-authorized-publish-command-plan.json"
$finalPackageReview = Read-JsonOrNull "artifacts\final-release\final-package-review-bundle.json"

$recordPath = "artifacts/final-release/post-publish-verification-record.json"
$templatePath = "artifacts/final-release/post-publish-verification-record-template.json"
$validationPath = "artifacts/final-release/post-publish-verification-validation.json"
$validateCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishVerificationRecord.ps1 -InputPath $recordPath -RequireExistingLog -FailOnNotProof"

$postPublishRequiredEvidence = @(
  "selectedChannel",
  "channelSourceUri",
  "publishedPackageUrl",
  "managedPackageUrl",
  "runtimePackageUrl",
  "managedNupkgSha256",
  "runtimeNupkgSha256",
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
  "hostMetadata"
)

$targetChannelPlaceholders = @(
  [pscustomobject]@{ field = "selectedChannel"; example = "nuget.org or github-packages"; boundary = "Local feed is not public channel proof." },
  [pscustomobject]@{ field = "channelSourceUri"; example = "https://api.nuget.org/v3/index.json or owner-selected source"; boundary = "The record must point to the real channel used by clean restore." },
  [pscustomobject]@{ field = "publishedPackageUrl"; example = "<managed/runtime package URL from selected channel>"; boundary = "A local nupkg path is not a public package URL." }
)

$requiredConsumerIdentity = @(
  "cleanConsumerRootOutsideRepository",
  "cleanConsumerRoot outside repository",
  "consumerProjectName",
  "consumerProjectPath ending in .csproj",
  "consumerProjectPath",
  "noProjectReference",
  "noLocalPackageSource",
  "noLocalNupkgPackageReference",
  "no ProjectReference",
  "restore/build/smoke commands sourced from selected channel only"
)

$requiredPackageHashes = @(
  [pscustomobject]@{ field = "managedPackageUrl"; source = "managed package URL from selected channel"; boundary = "A local path is not a selected-channel package URL." },
  [pscustomobject]@{ field = "runtimePackageUrl"; source = "runtime package URL from selected channel"; boundary = "A local path is not a selected-channel runtime package URL." },
  [pscustomobject]@{ field = "managedNupkgSha256"; source = "downloaded managed package from selected channel"; boundary = "Final package review local hash cannot replace channel download hash." },
  [pscustomobject]@{ field = "runtimeNupkgSha256"; source = "downloaded runtime package from selected channel"; boundary = "Runtime package hash must match the published runtime package identity." }
)

$requiredHostMetadata = @(
  "ownerName",
  "machineName",
  "osDescription",
  "gpuName",
  "driverVersion",
  "cudaDriverSupportedRuntime",
  "cudaRuntimeVersion",
  "tensorRtRuntimeVersion",
  "tensorRtLine",
  "cudnnVersion"
)

$requiredStdoutStderrSummaries = @(
  [pscustomobject]@{ field = "stdoutSummary"; required = "reviewed restore/build/dependency/smoke stdout summary"; boundary = "Log hashes without reviewed stdout are incomplete." },
  [pscustomobject]@{ field = "stderrSummary"; required = "reviewed stderr summary or no-stderr-emitted"; boundary = "Use no-stderr-emitted only when stderr is actually empty." }
)

$requiredPostPublishLogs = @(
  [pscustomobject]@{ field = "restoreLogPath"; required = "restore log captured from clean consumer restore against selectedChannel."; boundary = "A restore command without a preserved log cannot close release issue." },
  [pscustomobject]@{ field = "nativeAssetListingSha256"; required = "SHA256 for native asset listing after clean consumer build/copy."; boundary = "Native asset presence must be traceable, not inferred from build success." },
  [pscustomobject]@{ field = "dependencyProbeLogPath"; required = "DependencyProbe log path from clean consumer execution."; boundary = "DependencyProbe is diagnostics only and still needs preserved evidence." },
  [pscustomobject]@{ field = "dependencyProbeLogSha256"; required = "SHA256 for dependency probe log."; boundary = "A probe pass without matching log hash is not auditable." },
  [pscustomobject]@{ field = "runtimeSmokeLogPath"; required = "Runtime smoke log path from compatible host clean consumer execution."; boundary = "Smoke status without preserved log is not proof." },
  [pscustomobject]@{ field = "runtimeSmokeLogSha256"; required = "SHA256 for runtime smoke log."; boundary = "runtimeSmokePassed=true requires a matching smoke log hash." },
  [pscustomobject]@{ field = "runtimeSmokePassed"; required = "True only after compatible-host smoke passes."; boundary = "blocked-by-cuda-driver is not smoke passed." },
  [pscustomobject]@{ field = "runtimeSmokeExitCode"; required = "0 for the compatible-host runtime smoke command."; boundary = "Non-zero smoke exit cannot close release issue." },
  [pscustomobject]@{ field = "hostMetadata"; required = "OS, GPU, driver, CUDA, TensorRT, cuDNN, owner and machine metadata."; boundary = "Smoke evidence without host metadata is not reproducible." }
)

$copyableExecutionOrder = @(
  "Review artifacts/final-release/release-owner-approval-input-validation.json, release-owner-decision-record.json, and owner-authorized-publish-command-plan.json before any manual publish outside this script.",
  "Copy-Item -LiteralPath $templatePath -Destination $recordPath",
  "Download managed/runtime packages from <selected-channel> and capture package URLs plus SHA256 hashes.",
  "New-Item -ItemType Directory <clean-consumer-root-outside-repo>; dotnet new console; dotnet add package <managed> --source <selected-channel>; dotnet add package <runtime> --source <selected-channel>",
  "dotnet restore <clean-consumer.csproj> --source <selected-channel> *> <restore.log>",
  "dotnet build <clean-consumer.csproj> -c Release --no-restore *> <build.log>",
  "dotnet run --project <clean-consumer.csproj> -- --dependency-probe --runtime-package-key $RuntimePackageKey *> <dependency-probe.log>",
  "dotnet run --project <clean-consumer.csproj> -- --runtime-package-key $RuntimePackageKey --smoke *> <runtime-smoke.log>",
  "Get-FileHash -LiteralPath <restore.log>,<native-asset-listing.log>,<dependency-probe.log>,<runtime-smoke.log> -Algorithm SHA256",
  "Fill $recordPath with real channel, package identity, host metadata, command capture, stdoutSummary, stderrSummary, and matching log hashes.",
  $validateCommand,
  "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseCandidateFreezeSummary.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseCandidateFreezeSummary.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-OwnerAuthorizedPublishCommandPlan.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerAuthorizedPublishCommandPlan.ps1"
)

$collectionSteps = @(
  New-CollectionStep -Id "confirm-owner-publication" -Order 1 -Command "Review owner approval, owner decision, owner command plan, selected channel, rollback plan, and credential handling." -RequiredEvidence "Explicit owner authorization before manual publication outside this script." -Boundary "This collection package never runs publish/upload/delete/delist/withdraw."
  New-CollectionStep -Id "copy-template" -Order 2 -Command "Copy-Item -LiteralPath $templatePath -Destination $recordPath" -RequiredEvidence "recordKind=post-publish-verification-record and templateOnly=false only after real publication." -Boundary "A copied template is not proof."
  New-CollectionStep -Id "capture-channel-identity" -Order 3 -Command "Download package files from <selected-channel>; capture package URLs and SHA256 hashes." -RequiredEvidence "Selected channel, package IDs, versions, URLs, downloaded nupkg SHA256." -Boundary "Local feed and final package review are not public channel proof."
  New-CollectionStep -Id "create-clean-consumer" -Order 4 -Command "Create a clean project outside the source repository and add packages from the selected channel only." -RequiredEvidence "Clean consumer project path and no ProjectReference." -Boundary "ProjectReference invalidates post-publish proof."
  New-CollectionStep -Id "restore-build" -Order 5 -Command "dotnet restore/build the clean consumer from selected channel and hash restore/build/native asset logs." -RequiredEvidence "Restore/build commands, restore log SHA256, native asset listing SHA256, nativeAssetsCopied=true." -Boundary "Build success alone is not runtime smoke proof."
  New-CollectionStep -Id "dependency-probe-and-smoke" -Order 6 -Command "Run dependency probe and runtime smoke with --runtime-package-key $RuntimePackageKey." -RequiredEvidence "dependencyProbePassed=true, runtimeSmokePassed=true, runtimeSmokeExitCode=0, smokeStatus=passed." -Boundary "DependencyProbe is diagnostics only; blocked-by-cuda-driver is not smoke passed."
  New-CollectionStep -Id "review-stdout-stderr" -Order 7 -Command "Review restore/build/probe/smoke logs and fill stdoutSummary plus stderrSummary." -RequiredEvidence "Non-empty stdoutSummary and stderrSummary or no-stderr-emitted when stderr is empty." -Boundary "Missing stdout/stderr summaries cannot close the release issue."
  New-CollectionStep -Id "validate-real-record" -Order 8 -Command $validateCommand -RequiredEvidence $validationPath -Boundary "Only validator-promoted real post-publish proof can close the release issue."
  New-CollectionStep -Id "refresh-close-readiness" -Order 9 -Command "Refresh release evidence, freeze summary, owner command plan, and validators." -RequiredEvidence "Release close readiness artifacts agree after real proof validates." -Boundary "Aggregate refresh cannot fabricate proof."
)

$safetyNotes = @(
  "This collection package does not publish packages.",
  "This collection package does not approve public release.",
  "This collection package is not post-publish verification proof.",
  "This collection package keeps canCloseReleaseIssue=false.",
  "Local feed, final-package-review-bundle, template, draft, example, runbook, collection bundle, collection package, and dependency-probe-only outputs cannot close the release issue.",
  "Only post-publish-verification-record.json validated with -RequireExistingLog -FailOnNotProof can close the release issue.",
  "blocked-by-cuda-driver is not smoke passed.",
  "No dotnet nuget push, GitHub Packages upload, GitHub Release upload, delete, delist, or withdraw is performed."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "post-publish-verification-collection-package"
  packageState = "blocked-real-publication-required"
  runtimePackageKey = $RuntimePackageKey
  performsPublish = $false
  approvesPublicRelease = $false
  isPostPublishVerificationProof = $false
  canCloseReleaseIssue = $false
  currentBackfillPlanState = [string](Get-PropertyOrDefault -Object $backfillPlan -Name "planState" -DefaultValue "missing-post-publish-verification-backfill-plan")
  currentBackfillStepCount = @((Get-PropertyOrDefault -Object $backfillPlan -Name "backfillSteps" -DefaultValue @())).Count
  currentValidationState = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "validationState" -DefaultValue "missing-post-publish-verification-validation")
  currentProofClassification = [string](Get-PropertyOrDefault -Object $postPublishValidation -Name "postPublishProofClassification" -DefaultValue "missing-post-publish-proof-classification")
  currentValidationIsPostPublishVerificationProof = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "isPostPublishVerificationProof" -DefaultValue $false)
  currentValidationCanCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $postPublishValidation -Name "canCloseReleaseIssue" -DefaultValue $false)
  currentFailedProofItemCount = [int](Get-PropertyOrDefault -Object $postPublishValidation -Name "failedProofItemCount" -DefaultValue -1)
  ownerAuthorizedPublishCommandPlanState = [string](Get-PropertyOrDefault -Object $ownerPlan -Name "planState" -DefaultValue "missing-owner-authorized-publish-command-plan")
  finalPackageReviewState = [string](Get-PropertyOrDefault -Object $finalPackageReview -Name "bundleState" -DefaultValue "missing-final-package-review-bundle")
  expectedArtifacts = [pscustomobject]@{
    templatePath = $templatePath
    realRecordPath = $recordPath
    validationPath = $validationPath
    collectionPackageJsonPath = "artifacts/final-release/post-publish-verification-collection-package.json"
    collectionPackageMarkdownPath = "artifacts/final-release/post-publish-verification-collection-package.md"
  }
  targetChannelPlaceholders = $targetChannelPlaceholders
  requiredConsumerIdentity = $requiredConsumerIdentity
  requiredPackageHashes = $requiredPackageHashes
  requiredHostMetadata = $requiredHostMetadata
  requiredStdoutStderrSummaries = $requiredStdoutStderrSummaries
  requiredPostPublishLogs = $requiredPostPublishLogs
  postPublishRequiredEvidence = $postPublishRequiredEvidence
  postPublishRequiredEvidenceCount = $postPublishRequiredEvidence.Count
  copyableExecutionOrder = $copyableExecutionOrder
  collectionSteps = $collectionSteps
  validationCommand = $validateCommand
  sourceEvidence = @(
    "artifacts/final-release/post-publish-verification-validation.json",
    "artifacts/final-release/post-publish-verification-backfill-plan.json",
    "artifacts/final-release/owner-authorized-publish-command-plan.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/release-owner-decision-record.json",
    "artifacts/final-release/final-package-review-bundle.json"
  )
  safetyNotes = $safetyNotes
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "post-publish-verification-collection-package.json"
$markdownPath = Join-Path $outputRoot "post-publish-verification-collection-package.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Post-Publish Verification Collection Package")
$lines.Add("")
$lines.Add("- package state: ``$($record.packageState)``")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- post-publish proof: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("This package is copyable owner guidance for collecting real post-publish clean-consumer proof after an authorized publication. It is not a package push, public release approval, post-publish proof, or release close approval.")
$lines.Add("")
$lines.Add("## Copyable Execution Order")
$lines.Add("")
foreach ($command in $copyableExecutionOrder) {
  $lines.Add('```powershell')
  $lines.Add($command)
  $lines.Add('```')
}
$lines.Add("")
$lines.Add("## Target Channel Placeholders")
$lines.Add("")
$lines.Add("| Field | Example | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($item in $targetChannelPlaceholders) {
  $lines.Add("| ``$($item.field)`` | $(ConvertTo-MarkdownCell $item.example) | $(ConvertTo-MarkdownCell $item.boundary) |")
}
$lines.Add("")
$lines.Add("## Required Consumer Identity")
$lines.Add("")
foreach ($item in $requiredConsumerIdentity) { $lines.Add("- $item") }
$lines.Add("")
$lines.Add("## Post-Publish Required Evidence")
$lines.Add("")
foreach ($field in $postPublishRequiredEvidence) { $lines.Add("- ``$field``") }
$lines.Add("")
$lines.Add("## Required Package Hashes")
$lines.Add("")
$lines.Add("| Field | Source | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($hash in $requiredPackageHashes) {
  $lines.Add("| ``$($hash.field)`` | $(ConvertTo-MarkdownCell $hash.source) | $(ConvertTo-MarkdownCell $hash.boundary) |")
}
$lines.Add("")
$lines.Add("## Required Host Metadata")
$lines.Add("")
foreach ($field in $requiredHostMetadata) { $lines.Add("- ``$field``") }
$lines.Add("")
$lines.Add("## Stdout/Stderr Review")
$lines.Add("")
$lines.Add("| Field | Required | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($summary in $requiredStdoutStderrSummaries) {
  $lines.Add("| ``$($summary.field)`` | $(ConvertTo-MarkdownCell $summary.required) | $(ConvertTo-MarkdownCell $summary.boundary) |")
}
$lines.Add("")
$lines.Add("## Required Logs And Runtime Smoke Evidence")
$lines.Add("")
$lines.Add("| Field | Required | Boundary |")
$lines.Add("| --- | --- | --- |")
foreach ($log in $requiredPostPublishLogs) {
  $lines.Add("| ``$($log.field)`` | $(ConvertTo-MarkdownCell $log.required) | $(ConvertTo-MarkdownCell $log.boundary) |")
}
$lines.Add("")
$lines.Add("## Collection Steps")
$lines.Add("")
$lines.Add("| Order | ID | Command | Required evidence | Boundary |")
$lines.Add("| --- | --- | --- | --- | --- |")
foreach ($step in $collectionSteps) {
  $lines.Add("| $($step.order) | ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.command) | $(ConvertTo-MarkdownCell $step.requiredEvidence) | $(ConvertTo-MarkdownCell $step.boundary) |")
}
$lines.Add("")
$lines.Add("## Safety Notes")
$lines.Add("")
foreach ($note in $safetyNotes) { $lines.Add("- $note") }
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish verification collection package written to $jsonPath"
Write-Host "Post-publish verification collection package written to $markdownPath"
Write-Host "PackageState=$($record.packageState) PerformsPublish=False IsPostPublishVerificationProof=False CanCloseReleaseIssue=False"
