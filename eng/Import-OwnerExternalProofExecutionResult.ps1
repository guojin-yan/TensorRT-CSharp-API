[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-external-proof-execution-result.input.json",
  [string[]]$AllowedEvidenceRoots = @(
    "artifacts\final-release",
    "artifacts\package-consumer",
    "artifacts\runtime-proof",
    "artifacts\user-acceptance",
    "artifacts\yolovision",
    "artifacts\smoke",
    "artifacts\logs",
    "artifacts\packages"
  ),
  [string]$OutputRoot,
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

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Resolve-RepositoryPath -Path $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Test-Placeholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text -like "<*>"
}

function Test-Sha256Text {
  param([AllowNull()][object]$Value)
  return ([string]$Value) -match "^[0-9a-fA-F]{64}$"
}

function Test-PathUnderAllowedEvidenceRoot {
  param([AllowNull()][object]$Path, [string[]]$AllowedRoots)

  $pathText = [string]$Path
  if (Test-Placeholder -Value $pathText) { return $false }

  $fullPath = [IO.Path]::GetFullPath((Resolve-RepositoryPath -Path $pathText))
  foreach ($root in $AllowedRoots) {
    $rootPath = [IO.Path]::GetFullPath((Resolve-RepositoryPath -Path $root))
    if (-not $rootPath.EndsWith([IO.Path]::DirectorySeparatorChar)) {
      $rootPath = $rootPath + [IO.Path]::DirectorySeparatorChar
    }

    if ($fullPath.StartsWith($rootPath, [StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }

  return $false
}

function Test-ForbiddenSubstituteText {
  param([AllowNull()][object]$Value)

  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) { return $false }

  return $text -match "(?i)(ProjectReference|local\s*feed|direct\s*\.nupkg|dependency-probe-only|build-only|template-only|blocked-by-cuda-driver|sidecar-only|precheck-only|skipped)"
}

function Test-ZeroExitCode {
  param([AllowNull()][object]$Value)

  return [string]::Equals(([string]$Value).Trim(), "0", [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-TrueLike {
  param([AllowNull()][object]$Value)

  if ($Value -is [bool]) { return [bool]$Value }
  $text = ([string]$Value).Trim()
  return [string]::Equals($text, "true", [System.StringComparison]::OrdinalIgnoreCase) -or
    [string]::Equals($text, "passed", [System.StringComparison]::OrdinalIgnoreCase) -or
    [string]::Equals($text, "success", [System.StringComparison]::OrdinalIgnoreCase) -or
    [string]::Equals($text, "0", [System.StringComparison]::OrdinalIgnoreCase)
}

function Get-NestedValue {
  param([AllowNull()][object]$Object, [string]$Path)
  if ($null -eq $Object) { return $null }
  $current = $Object
  foreach ($part in $Path.Split(".")) {
    if ($null -eq $current -or -not ($current.PSObject.Properties.Name -contains $part)) { return $null }
    $current = $current.PSObject.Properties[$part].Value
  }
  return $current
}

function Test-EvidenceFileAndHash {
  param([string]$Id, [AllowNull()][object]$Path, [AllowNull()][object]$Sha256, [string[]]$AllowedRoots)

  $pathText = [string]$Path
  $shaText = [string]$Sha256
  $pathProvided = -not (Test-Placeholder -Value $pathText)
  $sha256FormatValid = Test-Sha256Text -Value $shaText
  $underAllowedRoot = $pathProvided -and (Test-PathUnderAllowedEvidenceRoot -Path $pathText -AllowedRoots $AllowedRoots)
  $resolvedPath = if ($pathProvided) { Resolve-RepositoryPath -Path $pathText } else { "" }
  $fileExists = $pathProvided -and (Test-Path -LiteralPath $resolvedPath -PathType Leaf)
  $actualSha256 = ""
  $hashMatches = $false

  if ($fileExists -and $sha256FormatValid) {
    $actualSha256 = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash
    $hashMatches = $actualSha256.Equals($shaText, [StringComparison]::OrdinalIgnoreCase)
  }

  $ready = $pathProvided -and $sha256FormatValid -and $underAllowedRoot -and $fileExists -and $hashMatches
  $failureReasons = New-Object System.Collections.Generic.List[string]
  if (-not $pathProvided) { $failureReasons.Add("path-missing-or-placeholder") | Out-Null }
  if ($pathProvided -and -not $underAllowedRoot) { $failureReasons.Add("path-outside-allowed-evidence-root") | Out-Null }
  if (-not $sha256FormatValid) { $failureReasons.Add("sha256-missing-or-invalid") | Out-Null }
  if ($pathProvided -and -not $fileExists) { $failureReasons.Add("file-missing") | Out-Null }
  if ($fileExists -and $sha256FormatValid -and -not $hashMatches) { $failureReasons.Add("sha256-mismatch") | Out-Null }

  [pscustomobject]@{
    id = $Id
    path = $pathText
    resolvedPath = [string]$resolvedPath
    sha256 = $shaText
    actualSha256 = $actualSha256
    pathProvided = $pathProvided
    pathUnderAllowedEvidenceRoot = $underAllowedRoot
    sha256FormatValid = $sha256FormatValid
    fileExists = $fileExists
    hashMatches = $hashMatches
    existsAndHashMatches = $ready
    state = if ($ready) { "file-hash-ready" } else { "blocked-real-file-or-hash-required" }
    failureReasons = @($failureReasons.ToArray())
  }
}

function Get-ProvidedResultById {
  param([object[]]$ProvidedResults, [string]$ResultInputId)
  return $ProvidedResults | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "resultInputId" -DefaultValue "") -eq $ResultInputId } | Select-Object -First 1
}

function New-ResultImportItem {
  param([object]$TemplateResult, [AllowNull()][object]$ProvidedResult, [bool]$InputFileExists, [string[]]$AllowedRoots)

  $resultInputId = [string](Get-PropertyOrDefault -Object $TemplateResult -Name "resultInputId" -DefaultValue "unknown-result-input")
  $requiredFields = @(Get-PropertyOrDefault -Object $TemplateResult -Name "requiredResultFields" -DefaultValue @())
  $missingFields = New-Object System.Collections.Generic.List[string]
  foreach ($field in $requiredFields) {
    $value = Get-NestedValue -Object $ProvidedResult -Path ([string]$field)
    if (Test-Placeholder -Value $value) {
      $missingFields.Add([string]$field) | Out-Null
    }
  }

  $fileChecks = @(
    [pscustomobject]@{ id = "nupkg"; path = Get-NestedValue -Object $ProvidedResult -Path "packageIdentity.nupkgPath"; sha256 = Get-NestedValue -Object $ProvidedResult -Path "packageIdentity.nupkgSha256" }
    [pscustomobject]@{ id = "stdout"; path = Get-NestedValue -Object $ProvidedResult -Path "stdoutPath"; sha256 = Get-NestedValue -Object $ProvidedResult -Path "stdoutSha256" }
    [pscustomobject]@{ id = "stderr"; path = Get-NestedValue -Object $ProvidedResult -Path "stderrPath"; sha256 = Get-NestedValue -Object $ProvidedResult -Path "stderrSha256" }
    [pscustomobject]@{ id = "merged-transcript"; path = Get-NestedValue -Object $ProvidedResult -Path "mergedTranscriptPath"; sha256 = Get-NestedValue -Object $ProvidedResult -Path "mergedTranscriptSha256" }
    [pscustomobject]@{ id = "validator-output"; path = Get-NestedValue -Object $ProvidedResult -Path "validatorOutputPath"; sha256 = Get-NestedValue -Object $ProvidedResult -Path "validatorOutputSha256" }
  ) | ForEach-Object {
    Test-EvidenceFileAndHash -Id $_.id -Path $_.path -Sha256 $_.sha256 -AllowedRoots $AllowedRoots
  }
  $failedFileChecks = @($fileChecks | Where-Object { -not [bool]$_.existsAndHashMatches })
  $fileMissingCount = @($fileChecks | Where-Object { -not [bool]$_.fileExists }).Count
  $invalidSha256Count = @($fileChecks | Where-Object { -not [bool]$_.sha256FormatValid }).Count
  $hashMismatchCount = @($fileChecks | Where-Object { [bool]$_.fileExists -and [bool]$_.sha256FormatValid -and -not [bool]$_.hashMatches }).Count
  $outsideAllowedEvidenceRootCount = @($fileChecks | Where-Object { [bool]$_.pathProvided -and -not [bool]$_.pathUnderAllowedEvidenceRoot }).Count
  $provided = $InputFileExists -and $null -ne $ProvidedResult
  $proofLane = [string](Get-PropertyOrDefault -Object $TemplateResult -Name "proofLane" -DefaultValue "")
  $confirmationValue = Get-NestedValue -Object $ProvidedResult -Path "nonSubstituteConfirmations"
  $nonSubstituteConfirmations = if ($null -eq $confirmationValue) { @() } else { @($confirmationValue) }
  $forbiddenSubstituteFindings = New-Object System.Collections.Generic.List[string]
  foreach ($value in @(
      (Get-NestedValue -Object $ProvidedResult -Path "packageIdentity.packageSource"),
      (Get-NestedValue -Object $ProvidedResult -Path "packageIdentity.nupkgPath"),
      (Get-NestedValue -Object $ProvidedResult -Path "executedCommandLine"),
      (Get-NestedValue -Object $ProvidedResult -Path "workingDirectory"),
      ($nonSubstituteConfirmations -join "; ")
    )) {
    if (Test-ForbiddenSubstituteText -Value $value) {
      $forbiddenSubstituteFindings.Add([string]$value) | Out-Null
    }
  }

  $exitCodeZero = Test-ZeroExitCode -Value (Get-NestedValue -Object $ProvidedResult -Path "exitCode")
  $passedTrue = Test-TrueLike -Value (Get-NestedValue -Object $ProvidedResult -Path "passed")
  $runtimeExecutionFieldsReady = -not (Test-Placeholder -Value (Get-NestedValue -Object $ProvidedResult -Path "stdoutPath")) -and
    -not (Test-Placeholder -Value (Get-NestedValue -Object $ProvidedResult -Path "stderrPath")) -and
    -not (Test-Placeholder -Value (Get-NestedValue -Object $ProvidedResult -Path "mergedTranscriptPath"))
  $ownerReviewReady = -not (Test-Placeholder -Value (Get-NestedValue -Object $ProvidedResult -Path "ownerReviewer")) -and
    -not (Test-Placeholder -Value (Get-NestedValue -Object $ProvidedResult -Path "ownerReviewTimestampUtc"))
  $nonSubstituteConfirmationsReady = $nonSubstituteConfirmations.Count -ge 10 -and $forbiddenSubstituteFindings.Count -eq 0
  $ready = $provided -and $missingFields.Count -eq 0 -and $failedFileChecks.Count -eq 0 -and $exitCodeZero -and $passedTrue -and $runtimeExecutionFieldsReady -and $ownerReviewReady -and $nonSubstituteConfirmationsReady

  [pscustomobject]@{
    resultImportItemId = "$resultInputId-owner-external-proof-result-import"
    resultInputId = $resultInputId
    executionInputId = [string](Get-PropertyOrDefault -Object $TemplateResult -Name "executionInputId" -DefaultValue "")
    candidateId = [string](Get-PropertyOrDefault -Object $TemplateResult -Name "candidateId" -DefaultValue "")
    proofLane = $proofLane
    runtimePackageKey = [string](Get-PropertyOrDefault -Object $TemplateResult -Name "runtimePackageKey" -DefaultValue "")
    importItemState = if ($ready) { "owner-external-proof-execution-result-ready" } else { "blocked-owner-external-proof-execution-result-required" }
    ownerResultProvided = $provided
    requiredResultFields = $requiredFields
    missingResultFields = @($missingFields.ToArray())
    missingResultFieldCount = $missingFields.Count
    fileEvidenceChecks = @($fileChecks)
    failedFileEvidenceCheckCount = $failedFileChecks.Count
    fileMissingCount = $fileMissingCount
    invalidSha256Count = $invalidSha256Count
    hashMismatchCount = $hashMismatchCount
    outsideAllowedEvidenceRootCount = $outsideAllowedEvidenceRootCount
    exitCode = [string](Get-NestedValue -Object $ProvidedResult -Path "exitCode")
    exitCodeZero = $exitCodeZero
    passed = Get-NestedValue -Object $ProvidedResult -Path "passed"
    passedTrue = $passedTrue
    stdoutPath = [string](Get-NestedValue -Object $ProvidedResult -Path "stdoutPath")
    stderrPath = [string](Get-NestedValue -Object $ProvidedResult -Path "stderrPath")
    mergedTranscriptPath = [string](Get-NestedValue -Object $ProvidedResult -Path "mergedTranscriptPath")
    runtimeExecutionFieldsReady = $runtimeExecutionFieldsReady
    ownerReviewer = [string](Get-NestedValue -Object $ProvidedResult -Path "ownerReviewer")
    ownerReviewTimestampUtc = [string](Get-NestedValue -Object $ProvidedResult -Path "ownerReviewTimestampUtc")
    ownerReviewReady = $ownerReviewReady
    nonSubstituteConfirmationCount = $nonSubstituteConfirmations.Count
    nonSubstituteConfirmationsReady = $nonSubstituteConfirmationsReady
    forbiddenSubstituteFindingCount = $forbiddenSubstituteFindings.Count
    forbiddenSubstituteFindings = @($forbiddenSubstituteFindings.ToArray())
    readyForRealProofRecordImport = $ready
    canPromoteLaneResult = $false
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    nonProofBoundary = "Imported owner result metadata is not proof until all referenced files exist, hashes match, validators pass, and later real proof import validation promotes it."
  }
}

$template = Read-JsonOrNull "artifacts\final-release\owner-runtime-proof-result-input.template.json"
$bundle = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-bundle.json"
if ($null -eq $template) { throw "Missing owner-runtime-proof-result-input.template.json. Run Export-OwnerRuntimeProofResultInputTemplate.ps1 first." }
if ($null -eq $bundle) { throw "Missing owner-external-proof-execution-bundle.json. Run Export-OwnerExternalProofExecutionBundle.ps1 first." }

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
$inputFileExists = Test-Path -LiteralPath $resolvedInputPath -PathType Leaf
$providedRecord = if ($inputFileExists) { Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json } else { $null }
$providedResults = @(Get-PropertyOrDefault -Object $providedRecord -Name "resultInputs" -DefaultValue @())
if ($providedResults.Count -eq 0 -and $null -ne $providedRecord -and ($providedRecord.PSObject.Properties.Name -contains "resultInputId")) {
  $providedResults = @($providedRecord)
}

$templateResults = @(Get-PropertyOrDefault -Object $template -Name "resultInputs" -DefaultValue @())
$importItems = @($templateResults | ForEach-Object {
    $providedResult = Get-ProvidedResultById -ProvidedResults $providedResults -ResultInputId ([string](Get-PropertyOrDefault -Object $_ -Name "resultInputId" -DefaultValue ""))
    New-ResultImportItem -TemplateResult $_ -ProvidedResult $providedResult -InputFileExists $inputFileExists -AllowedRoots $AllowedEvidenceRoots
  })
$blockedItems = @($importItems | Where-Object { [string]$_.importItemState -eq "blocked-owner-external-proof-execution-result-required" })
$readyItems = @($importItems | Where-Object { [bool]$_.readyForRealProofRecordImport })
$promotableItems = @($importItems | Where-Object { [bool]$_.canPromoteLaneResult })
$runtimeLane = $importItems | Where-Object { [string]$_.proofLane -eq "package-consumer-runtime" } | Select-Object -First 1
$postPublishLane = $importItems | Where-Object { [string]$_.proofLane -eq "post-publish-verification" } | Select-Object -First 1
$releaseCloseLane = $importItems | Where-Object { [string]$_.proofLane -eq "release-close-owner-input" } | Select-Object -First 1
$strictCloseLane = $importItems | Where-Object { [string]$_.proofLane -eq "strict-close-validation" } | Select-Object -First 1
$laneSummaries = @($importItems | ForEach-Object {
    [pscustomobject]@{
      proofLane = [string]$_.proofLane
      runtimePackageKey = [string]$_.runtimePackageKey
      importItemState = [string]$_.importItemState
      readyForRealProofRecordImport = [bool]$_.readyForRealProofRecordImport
      canPromoteLaneResult = [bool]$_.canPromoteLaneResult
      missingResultFieldCount = [int]$_.missingResultFieldCount
      failedFileEvidenceCheckCount = [int]$_.failedFileEvidenceCheckCount
      fileMissingCount = [int]$_.fileMissingCount
      invalidSha256Count = [int]$_.invalidSha256Count
      hashMismatchCount = [int]$_.hashMismatchCount
      outsideAllowedEvidenceRootCount = [int]$_.outsideAllowedEvidenceRootCount
      forbiddenSubstituteFindingCount = [int]$_.forbiddenSubstituteFindingCount
      exitCodeZero = [bool]$_.exitCodeZero
      passedTrue = [bool]$_.passedTrue
      runtimeExecutionFieldsReady = [bool]$_.runtimeExecutionFieldsReady
      ownerReviewReady = [bool]$_.ownerReviewReady
      nonSubstituteConfirmationsReady = [bool]$_.nonSubstituteConfirmationsReady
    }
  })
$missingRealEvidenceCount = ($importItems | ForEach-Object { [int]$_.missingResultFieldCount + [int]$_.failedFileEvidenceCheckCount } | Measure-Object -Sum).Sum
if ($null -eq $missingRealEvidenceCount) { $missingRealEvidenceCount = 0 }
$fileMissingCount = ($importItems | ForEach-Object { [int]$_.fileMissingCount } | Measure-Object -Sum).Sum
if ($null -eq $fileMissingCount) { $fileMissingCount = 0 }
$invalidSha256Count = ($importItems | ForEach-Object { [int]$_.invalidSha256Count } | Measure-Object -Sum).Sum
if ($null -eq $invalidSha256Count) { $invalidSha256Count = 0 }
$hashMismatchCount = ($importItems | ForEach-Object { [int]$_.hashMismatchCount } | Measure-Object -Sum).Sum
if ($null -eq $hashMismatchCount) { $hashMismatchCount = 0 }
$outsideAllowedEvidenceRootCount = ($importItems | ForEach-Object { [int]$_.outsideAllowedEvidenceRootCount } | Measure-Object -Sum).Sum
if ($null -eq $outsideAllowedEvidenceRootCount) { $outsideAllowedEvidenceRootCount = 0 }
$forbiddenSubstituteFindingCount = ($importItems | ForEach-Object { [int]$_.forbiddenSubstituteFindingCount } | Measure-Object -Sum).Sum
if ($null -eq $forbiddenSubstituteFindingCount) { $forbiddenSubstituteFindingCount = 0 }
$forbiddenSubstituteMarkers = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "build-only",
  "dry-run",
  "candidate",
  "dashboard",
  "blocked-by-driver",
  "blocked-by-cuda-driver",
  "dependency-probe-only",
  "template-only",
  "precheck-only",
  "sidecar-only"
)
$laneReadinessSummary = @($laneSummaries | ForEach-Object {
    $blockedReasons = New-Object System.Collections.Generic.List[string]
    if (-not [bool]$_.readyForRealProofRecordImport) { $blockedReasons.Add("not-ready-for-real-proof-record-import") | Out-Null }
    if ([int]$_.missingResultFieldCount -gt 0) { $blockedReasons.Add("missing-result-fields=$($_.missingResultFieldCount)") | Out-Null }
    if ([int]$_.failedFileEvidenceCheckCount -gt 0) { $blockedReasons.Add("failed-file-evidence=$($_.failedFileEvidenceCheckCount)") | Out-Null }
    if ([int]$_.fileMissingCount -gt 0) { $blockedReasons.Add("file-missing=$($_.fileMissingCount)") | Out-Null }
    if ([int]$_.invalidSha256Count -gt 0) { $blockedReasons.Add("invalid-sha256=$($_.invalidSha256Count)") | Out-Null }
    if ([int]$_.hashMismatchCount -gt 0) { $blockedReasons.Add("hash-mismatch=$($_.hashMismatchCount)") | Out-Null }
    if ([int]$_.outsideAllowedEvidenceRootCount -gt 0) { $blockedReasons.Add("outside-allowed-root=$($_.outsideAllowedEvidenceRootCount)") | Out-Null }
    if ([int]$_.forbiddenSubstituteFindingCount -gt 0) { $blockedReasons.Add("forbidden-substitute=$($_.forbiddenSubstituteFindingCount)") | Out-Null }
    if (-not [bool]$_.exitCodeZero) { $blockedReasons.Add("exit-code-not-zero-or-missing") | Out-Null }
    if (-not [bool]$_.passedTrue) { $blockedReasons.Add("passed-flag-not-true-or-missing") | Out-Null }
    if (-not [bool]$_.runtimeExecutionFieldsReady) { $blockedReasons.Add("stdout-stderr-transcript-paths-required") | Out-Null }
    if (-not [bool]$_.ownerReviewReady) { $blockedReasons.Add("owner-review-required") | Out-Null }
    if (-not [bool]$_.nonSubstituteConfirmationsReady) { $blockedReasons.Add("non-substitute-confirmations-required") | Out-Null }

    [pscustomobject]@{
      proofLane = [string]$_.proofLane
      laneState = [string]$_.importItemState
      readyForStrictValidator = [bool]$_.readyForRealProofRecordImport
      stillBlocked = -not [bool]$_.readyForRealProofRecordImport
      blockedReasons = @($blockedReasons.ToArray())
      strictValidatorInputOnly = $true
      canPromoteLaneResult = $false
    }
  })
$summary = [pscustomobject]@{
  readyForStrictValidatorLaneCount = $readyItems.Count
  blockedLaneCount = $blockedItems.Count
  promotableLaneCount = $promotableItems.Count
  readyForStrictValidatorLanes = @($laneReadinessSummary | Where-Object { [bool]$_.readyForStrictValidator } | ForEach-Object { [string]$_.proofLane })
  blockedLanes = @($laneReadinessSummary | Where-Object { [bool]$_.stillBlocked } | ForEach-Object { [string]$_.proofLane })
  strictValidatorInputOnly = $true
  proofPromotionAllowed = $false
  publishAllowed = $false
  releaseCloseAllowed = $false
}

$record = [pscustomobject]@{
  recordKind = "owner-external-proof-execution-result-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  inputFileExists = $inputFileExists
  importState = if ($readyItems.Count -eq $templateResults.Count -and $templateResults.Count -gt 0) { "owner-external-proof-execution-result-ready" } else { "blocked-owner-external-proof-execution-result-required" }
  executionBundleState = [string](Get-PropertyOrDefault -Object $bundle -Name "bundleState" -DefaultValue "missing-owner-external-proof-execution-bundle")
  resultImportItemCount = $importItems.Count
  blockedResultImportItemCount = $blockedItems.Count
  readyResultImportItemCount = $readyItems.Count
  promotableResultImportItemCount = $promotableItems.Count
  ownerExternalProofResultLaneCount = $importItems.Count
  ownerExternalProofResultBlockedLaneCount = $blockedItems.Count
  ownerExternalProofResultReadyLaneCount = $readyItems.Count
  ownerExternalProofResultPromotableLaneCount = $promotableItems.Count
  missingRealEvidenceCount = [int]$missingRealEvidenceCount
  fileMissingCount = [int]$fileMissingCount
  invalidSha256Count = [int]$invalidSha256Count
  hashMismatchCount = [int]$hashMismatchCount
  outsideAllowedEvidenceRootCount = [int]$outsideAllowedEvidenceRootCount
  forbiddenSubstituteFindingCount = [int]$forbiddenSubstituteFindingCount
  allowedEvidenceRoots = [string[]]$AllowedEvidenceRoots
  forbiddenSubstituteMarkers = $forbiddenSubstituteMarkers
  summary = $summary
  laneReadinessSummary = $laneReadinessSummary
  laneSummaries = $laneSummaries
  packageConsumerRuntimeLaneReady = [bool](Get-PropertyOrDefault -Object $runtimeLane -Name "readyForRealProofRecordImport" -DefaultValue $false)
  postPublishVerificationLaneReady = [bool](Get-PropertyOrDefault -Object $postPublishLane -Name "readyForRealProofRecordImport" -DefaultValue $false)
  releaseCloseOwnerInputLaneReady = [bool](Get-PropertyOrDefault -Object $releaseCloseLane -Name "readyForRealProofRecordImport" -DefaultValue $false)
  strictCloseValidationLaneReady = [bool](Get-PropertyOrDefault -Object $strictCloseLane -Name "readyForRealProofRecordImport" -DefaultValue $false)
  resultImportItems = $importItems
  sourceArtifacts = @(
    "artifacts/final-release/owner-runtime-proof-result-input.template.json",
    "artifacts/final-release/owner-external-proof-execution-bundle.json",
    $InputPath
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Owner external proof execution result import is a strict import surface. Missing input, placeholder fields, missing files, or mismatched hashes keep it blocked and non-proof."
}

$artifactRoot = if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  Join-Path $RepositoryRoot "artifacts\final-release"
}
else {
  Resolve-RepositoryPath -Path $OutputRoot
}
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-external-proof-execution-result-import.json"
$markdownPath = Join-Path $artifactRoot "owner-external-proof-execution-result-import.md"
$record | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner External Proof Execution Result Import")
$lines.Add("")
$lines.Add("| 项目 | 当前值 |")
$lines.Add("|---|---|")
$lines.Add("| importState | ``$($record.importState)`` |")
$lines.Add("| inputFileExists | ``$($record.inputFileExists)`` |")
$lines.Add("| resultImportItemCount | ``$($record.resultImportItemCount)`` |")
$lines.Add("| blockedResultImportItemCount | ``$($record.blockedResultImportItemCount)`` |")
$lines.Add("| readyResultImportItemCount | ``$($record.readyResultImportItemCount)`` |")
$lines.Add("| promotableResultImportItemCount | ``$($record.promotableResultImportItemCount)`` |")
$lines.Add("| missingRealEvidenceCount | ``$($record.missingRealEvidenceCount)`` |")
$lines.Add("| fileMissingCount | ``$($record.fileMissingCount)`` |")
$lines.Add("| invalidSha256Count | ``$($record.invalidSha256Count)`` |")
$lines.Add("| hashMismatchCount | ``$($record.hashMismatchCount)`` |")
$lines.Add("| outsideAllowedEvidenceRootCount | ``$($record.outsideAllowedEvidenceRootCount)`` |")
$lines.Add("| forbiddenSubstituteFindingCount | ``$($record.forbiddenSubstituteFindingCount)`` |")
$lines.Add("| readyForStrictValidatorLaneCount | ``$($record.summary.readyForStrictValidatorLaneCount)`` |")
$lines.Add("| blockedLaneCount | ``$($record.summary.blockedLaneCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Lane Readiness Summary")
$lines.Add("")
$lines.Add("| Lane | Ready For Strict Validator | Blocked Reasons |")
$lines.Add("|---|---:|---|")
foreach ($lane in $laneReadinessSummary) {
  $lines.Add("| $($lane.proofLane) | ``$($lane.readyForStrictValidator)`` | $((@($lane.blockedReasons) -join '; ').Replace('|', '\|')) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.safetyBoundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external proof execution result import written to $jsonPath"
Write-Host "Owner external proof execution result import markdown written to $markdownPath"
Write-Host "ImportState=$($record.importState) Items=$($record.resultImportItemCount) Blocked=$($record.blockedResultImportItemCount) MissingRealEvidence=$($record.missingRealEvidenceCount)"
