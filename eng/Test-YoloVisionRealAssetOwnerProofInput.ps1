[CmdletBinding()]
param(
  [string]$InputPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$RequireExistingFiles
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = "artifacts/user-acceptance"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return @()
  }

  if ($Value -is [System.Array]) {
    return @($Value)
  }

  return @($Value)
}

function Test-IsPlaceholder {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or
    $text.Equals("owner-required", [StringComparison]::OrdinalIgnoreCase) -or
    $text.Equals("owner-required-or-no-stderr", [StringComparison]::OrdinalIgnoreCase) -or
    $text -like "<*>"
}

function Test-Sha256Format {
  param([AllowNull()][object]$Value)

  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-DateTimeOffsetFormat {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(([string]$Value).Trim(), [ref]$parsed)
}

function Test-PositiveInt64 {
  param([AllowNull()][object]$Value)

  if (Test-IsPlaceholder -Value $Value) {
    return $false
  }

  $parsed = [int64]0
  return [int64]::TryParse(([string]$Value).Trim(), [ref]$parsed) -and $parsed -gt 0
}

function Test-FileExistsIfRequired {
  param([AllowNull()][object]$Path)

  if (-not $RequireExistingFiles.IsPresent) {
    return $true
  }

  if (Test-IsPlaceholder -Value $Path) {
    return $false
  }

  return Test-Path -LiteralPath (Resolve-RepositoryPath -Path ([string]$Path)) -PathType Leaf
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Add-ValidationItem {
  param(
    [System.Collections.Generic.List[object]]$Items,
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  $Items.Add((New-ValidationItem -Id $Id -Passed $Passed -Severity $Severity -Detail $Detail)) | Out-Null
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "YoloVision real asset owner proof input was not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$proofClassification = [string](Get-PropertyOrDefault -Object $record -Name "proofClassification" -DefaultValue "")
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)
$canPromoteRealModelRuntimeDeclared = [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRealModelRuntime" -DefaultValue $true)
$canPromotePackageConsumerRuntime = [bool](Get-PropertyOrDefault -Object $record -Name "canPromotePackageConsumerRuntime" -DefaultValue $true)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)
$cases = @(ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "cases" -DefaultValue @()))
$globalEvidence = Get-PropertyOrDefault -Object $record -Name "requiredGlobalEvidence" -DefaultValue $null
$hostMetadata = Get-PropertyOrDefault -Object $globalEvidence -Name "hostMetadata" -DefaultValue $null
$packageMetadata = Get-PropertyOrDefault -Object $globalEvidence -Name "packageMetadata" -DefaultValue $null
$globalOwnerReview = Get-PropertyOrDefault -Object $globalEvidence -Name "ownerReview" -DefaultValue $null

Add-ValidationItem $items "record-kind" ($recordKind -eq "yolovision-real-asset-owner-proof-input") "blocker" "recordKind must be yolovision-real-asset-owner-proof-input."
Add-ValidationItem $items "no-publish-or-package-proof" (-not $performsPublish -and -not $canPublishPublicly -and -not $canPromotePackageConsumerRuntime -and -not $canCloseReleaseIssue) "blocker" "Owner proof input validation must not publish, close release, or promote package-consumer-runtime."
Add-ValidationItem $items "proof-classification-known" ($proofClassification -in @("template-only", "owner-input-candidate", "real-model-runtime")) "blocker" "proofClassification must be template-only, owner-input-candidate, or real-model-runtime."
Add-ValidationItem $items "case-count" ($cases.Count -eq 6) "blocker" "Owner proof input must contain exactly six YOLOv8n task cases, including semantic segmentation."

foreach ($field in @("hostOs", "hostMachineId", "osArchitecture", "gpuName", "gpuComputeCapability", "driverVersion", "cudaDriverVersion", "cudaRuntimeVersion", "tensorRtVersion", "tensorRtLine", "cudnnVersion")) {
  $value = Get-PropertyOrDefault -Object $hostMetadata -Name $field -DefaultValue ""
  Add-ValidationItem $items "global-host-$field" (-not (Test-IsPlaceholder -Value $value)) "owner-action-required" "hostMetadata.$field must be real non-placeholder owner input."
}

foreach ($field in @("packageSource", "packageChannel", "runtimePackageVersion", "runtimePackageKey")) {
  $value = Get-PropertyOrDefault -Object $packageMetadata -Name $field -DefaultValue ""
  Add-ValidationItem $items "global-package-$field" (-not (Test-IsPlaceholder -Value $value)) "owner-action-required" "packageMetadata.$field must be real non-placeholder owner input."
}

foreach ($field in @("managedPackageSha256", "nativeBridgeSha256", "runtimePackageSha256")) {
  $value = Get-PropertyOrDefault -Object $packageMetadata -Name $field -DefaultValue ""
  Add-ValidationItem $items "global-package-$field" (Test-Sha256Format -Value $value) "owner-action-required" "packageMetadata.$field must be a 64-character SHA256 hash."
}

foreach ($field in @("ownerReviewer", "ownerAcceptanceDecision", "ownerAcceptanceNotes")) {
  $value = Get-PropertyOrDefault -Object $globalOwnerReview -Name $field -DefaultValue ""
  Add-ValidationItem $items "global-owner-$field" (-not (Test-IsPlaceholder -Value $value)) "owner-action-required" "ownerReview.$field must be real non-placeholder owner input."
}
$ownerReviewedAtUtcValue = Get-PropertyOrDefault -Object $globalOwnerReview -Name "ownerReviewedAtUtc" -DefaultValue ""
Add-ValidationItem $items "global-owner-ownerReviewedAtUtc" (Test-DateTimeOffsetFormat -Value $ownerReviewedAtUtcValue) "owner-action-required" "ownerReview.ownerReviewedAtUtc must be parseable."

$caseReadyStates = New-Object System.Collections.Generic.List[object]
foreach ($case in $cases) {
  $caseId = [string](Get-PropertyOrDefault -Object $case -Name "caseId" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($caseId)) {
    $caseId = "unknown"
  }

  $task = [string](Get-PropertyOrDefault -Object $case -Name "task" -DefaultValue "")
  $model = Get-PropertyOrDefault -Object $case -Name "model" -DefaultValue $null
  $labels = Get-PropertyOrDefault -Object $case -Name "labels" -DefaultValue $null
  $input = Get-PropertyOrDefault -Object $case -Name "input" -DefaultValue $null
  $tensorRtExec = Get-PropertyOrDefault -Object $case -Name "tensorRtExec" -DefaultValue $null
  $yoloVision = Get-PropertyOrDefault -Object $case -Name "yoloVision" -DefaultValue $null
  $articleEvidence = Get-PropertyOrDefault -Object $case -Name "articleEvidence" -DefaultValue $null
  $ownerReview = Get-PropertyOrDefault -Object $case -Name "ownerReview" -DefaultValue $null
  $expectedLines = @(ConvertTo-Array (Get-PropertyOrDefault -Object $yoloVision -Name "expectedEvidenceLines" -DefaultValue @()))

  Add-ValidationItem $items "case-$caseId-task-supported" ($task -in @("det", "seg", "pose", "obb", "cls", "sem")) "blocker" "$caseId task must be one of det, seg, pose, obb, cls, or sem."
  Add-ValidationItem $items "case-$caseId-no-package-promotion" (-not [bool](Get-PropertyOrDefault -Object $case -Name "canPromotePackageConsumerRuntime" -DefaultValue $true)) "blocker" "$caseId must not promote package-consumer-runtime."
  Add-ValidationItem $items "case-$caseId-expected-passed" (@($expectedLines | Where-Object { ([string]$_).Contains("YoloVision Passed=True", [StringComparison]::Ordinal) }).Count -gt 0) "blocker" "$caseId expectedEvidenceLines must contain YoloVision Passed=True."

  foreach ($entry in @(
      @{ id = "model-source-url"; object = $model; name = "sourceUrl" },
      @{ id = "model-license"; object = $model; name = "license" },
      @{ id = "model-onnx-path"; object = $model; name = "onnxPath" },
      @{ id = "model-export-command"; object = $model; name = "exportCommand" },
      @{ id = "labels-path"; object = $labels; name = "path" },
      @{ id = "labels-license"; object = $labels; name = "license" },
      @{ id = "labels-class-count"; object = $labels; name = "classCount" },
      @{ id = "input-image-path"; object = $input; name = "imagePath" },
      @{ id = "input-image-license"; object = $input; name = "imageLicense" },
      @{ id = "input-preprocessed-path"; object = $input; name = "preprocessedTensorPath" },
      @{ id = "input-shape"; object = $input; name = "inputShape" },
      @{ id = "input-preprocess-contract"; object = $input; name = "preprocessContract" },
      @{ id = "trtexec-command"; object = $tensorRtExec; name = "buildCommand" },
      @{ id = "trtexec-report-path"; object = $tensorRtExec; name = "reportPath" },
      @{ id = "trtexec-stdout-log-path"; object = $tensorRtExec; name = "stdoutLogPath" },
      @{ id = "trtexec-stderr-log-path"; object = $tensorRtExec; name = "stderrLogPath" },
      @{ id = "trtexec-engine-path"; object = $tensorRtExec; name = "enginePath" },
      @{ id = "yolovision-command"; object = $yoloVision; name = "runCommand" },
      @{ id = "yolovision-run-log-path"; object = $yoloVision; name = "runLogPath" },
      @{ id = "yolovision-stdout-log-path"; object = $yoloVision; name = "stdoutLogPath" },
      @{ id = "yolovision-stderr-log-path"; object = $yoloVision; name = "stderrLogPath" },
      @{ id = "yolovision-output-json-path"; object = $yoloVision; name = "outputJsonPath" },
      @{ id = "stdout-summary"; object = $yoloVision; name = "stdoutSummary" },
      @{ id = "stderr-summary"; object = $yoloVision; name = "stderrSummary" },
      @{ id = "article-readiness"; object = $articleEvidence; name = "readiness" },
      @{ id = "article-status"; object = $articleEvidence; name = "articleStatus" },
      @{ id = "article-proof-boundary"; object = $articleEvidence; name = "proofBoundary" },
      @{ id = "owner-reviewer"; object = $ownerReview; name = "reviewer" },
      @{ id = "owner-acceptance-decision"; object = $ownerReview; name = "acceptanceDecision" },
      @{ id = "owner-notes"; object = $ownerReview; name = "notes" }
    )) {
    $value = Get-PropertyOrDefault -Object $entry.object -Name $entry.name -DefaultValue ""
    Add-ValidationItem $items "case-$caseId-$($entry.id)" (-not (Test-IsPlaceholder -Value $value)) "owner-action-required" "$caseId $($entry.name) must be real non-placeholder owner input."
  }

  foreach ($entry in @(
      @{ id = "model-sha256"; object = $model; name = "sha256" },
      @{ id = "onnx-sha256"; object = $model; name = "onnxSha256" },
      @{ id = "labels-sha256"; object = $labels; name = "sha256" },
      @{ id = "image-sha256"; object = $input; name = "imageSha256" },
      @{ id = "preprocessed-tensor-sha256"; object = $input; name = "preprocessedTensorSha256" },
      @{ id = "trtexec-report-sha256"; object = $tensorRtExec; name = "reportSha256" },
      @{ id = "trtexec-stdout-log-sha256"; object = $tensorRtExec; name = "stdoutLogSha256" },
      @{ id = "trtexec-stderr-log-sha256"; object = $tensorRtExec; name = "stderrLogSha256" },
      @{ id = "engine-sha256"; object = $tensorRtExec; name = "engineSha256" },
      @{ id = "run-log-sha256"; object = $yoloVision; name = "runLogSha256" },
      @{ id = "yolovision-stdout-log-sha256"; object = $yoloVision; name = "stdoutLogSha256" },
      @{ id = "yolovision-stderr-log-sha256"; object = $yoloVision; name = "stderrLogSha256" },
      @{ id = "output-json-sha256"; object = $yoloVision; name = "outputJsonSha256" }
    )) {
    $value = Get-PropertyOrDefault -Object $entry.object -Name $entry.name -DefaultValue ""
    $hashReady = if ([string]$entry.name -like "stderr*") {
      (([string]$value).Equals("no-stderr", [StringComparison]::OrdinalIgnoreCase)) -or (Test-Sha256Format -Value $value)
    }
    else {
      Test-Sha256Format -Value $value
    }
    Add-ValidationItem $items "case-$caseId-$($entry.id)" $hashReady "owner-action-required" "$caseId $($entry.name) must be a 64-character SHA256 hash, or no-stderr for stderr logs."
  }

  Add-ValidationItem $items "case-$caseId-preprocessed-element-count" (Test-PositiveInt64 -Value (Get-PropertyOrDefault -Object $input -Name "preprocessedTensorElementCount" -DefaultValue "")) "owner-action-required" "$caseId preprocessedTensorElementCount must be > 0."
  Add-ValidationItem $items "case-$caseId-reviewed-at-utc" (Test-DateTimeOffsetFormat -Value (Get-PropertyOrDefault -Object $ownerReview -Name "reviewedAtUtc" -DefaultValue "")) "owner-action-required" "$caseId ownerReview.reviewedAtUtc must be parseable."
  Add-ValidationItem $items "case-$caseId-accepted-for-candidate" ([bool](Get-PropertyOrDefault -Object $ownerReview -Name "acceptedForRealModelRuntimeCandidate" -DefaultValue $false)) "owner-action-required" "$caseId ownerReview.acceptedForRealModelRuntimeCandidate must be true before candidate projection can be ready."
  Add-ValidationItem $items "case-$caseId-smoke-passed" ([bool](Get-PropertyOrDefault -Object $case -Name "isSmokePassed" -DefaultValue $false)) "owner-action-required" "$caseId isSmokePassed must be true before real-model-runtime candidate projection can be ready."

  foreach ($pathEntry in @(
      @{ id = "model-file-exists"; object = $model; name = "onnxPath" },
      @{ id = "labels-file-exists"; object = $labels; name = "path" },
      @{ id = "input-image-file-exists"; object = $input; name = "imagePath" },
      @{ id = "preprocessed-file-exists"; object = $input; name = "preprocessedTensorPath" },
      @{ id = "trtexec-report-file-exists"; object = $tensorRtExec; name = "reportPath" },
      @{ id = "trtexec-stdout-log-file-exists"; object = $tensorRtExec; name = "stdoutLogPath" },
      @{ id = "trtexec-stderr-log-file-exists"; object = $tensorRtExec; name = "stderrLogPath" },
      @{ id = "engine-file-exists"; object = $tensorRtExec; name = "enginePath" },
      @{ id = "run-log-file-exists"; object = $yoloVision; name = "runLogPath" },
      @{ id = "yolovision-stdout-log-file-exists"; object = $yoloVision; name = "stdoutLogPath" },
      @{ id = "yolovision-stderr-log-file-exists"; object = $yoloVision; name = "stderrLogPath" },
      @{ id = "output-json-file-exists"; object = $yoloVision; name = "outputJsonPath" }
    )) {
    $pathValue = Get-PropertyOrDefault -Object $pathEntry.object -Name $pathEntry.name -DefaultValue ""
    Add-ValidationItem $items "case-$caseId-$($pathEntry.id)" (Test-FileExistsIfRequired -Path $pathValue) "owner-action-required" "$caseId $($pathEntry.name) must exist when -RequireExistingFiles is used."
  }
}

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedOwnerActions = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "owner-action-required" })
$candidateReadyForRealModelRuntime = $failedBlockers.Count -eq 0 -and $failedOwnerActions.Count -eq 0 -and -not $canPromotePackageConsumerRuntime

if ($failedBlockers.Count -gt 0) {
  $validationState = "invalid"
}
elseif ($candidateReadyForRealModelRuntime) {
  $validationState = "candidate-ready-for-real-model-runtime"
}
else {
  $validationState = "owner-action-required"
}

$outputRootResolved = Resolve-RepositoryPath -Path $OutputRoot
New-Item -ItemType Directory -Path $outputRootResolved -Force | Out-Null
$jsonPath = Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-input-validation.json"
$markdownPath = Join-Path $outputRootResolved "yolovision-real-asset-owner-proof-input-validation.md"

$summary = [pscustomobject]@{
  recordKind = "yolovision-real-asset-owner-proof-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  proofClassification = $proofClassification
  caseCount = $cases.Count
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequiredCount = $failedOwnerActions.Count
  candidateReadyForRealModelRuntime = $candidateReadyForRealModelRuntime
  performsPublish = $false
  canPublishPublicly = $false
  canPromotePackageConsumerRuntime = $false
  requireExistingFiles = $RequireExistingFiles.IsPresent
  proofBoundary = "owner input validation only; not package-consumer-runtime proof; not post-publish proof"
  validationItems = $validationItems
}

$summary | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# YoloVision Real Asset Owner Proof Input Validation

| Field | Value |
| --- | --- |
| validationState | ``$($summary.validationState)`` |
| caseCount | ``$($summary.caseCount)`` |
| failedBlockerCount | ``$($summary.failedBlockerCount)`` |
| ownerActionRequiredCount | ``$($summary.ownerActionRequiredCount)`` |
| candidateReadyForRealModelRuntime | ``$($summary.candidateReadyForRealModelRuntime)`` |
| canPromotePackageConsumerRuntime | ``$($summary.canPromotePackageConsumerRuntime)`` |

## Validation Items

| ID | Passed | Severity | Detail |
| --- | ---: | --- | --- |
$($rows -join "`r`n")

## Boundary

$($summary.proofBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "YoloVision real asset owner proof input validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) OwnerActionRequired=$($failedOwnerActions.Count) CandidateReadyForRealModelRuntime=$candidateReadyForRealModelRuntime"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
