[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

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

function ConvertTo-Array {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-StringArray {
  param([AllowNull()][object]$Value)
  return @(ConvertTo-Array $Value | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-ContractSurface {
  param(
    [string]$Id,
    [string]$Title,
    [string]$ArtifactPath,
    [string]$RecordKind,
    [string]$State,
    [string[]]$FieldPaths,
    [string[]]$Aliases,
    [string[]]$RequiredCanonicalFields
  )

  $fieldSet = New-Object System.Collections.Generic.HashSet[string] ([StringComparer]::OrdinalIgnoreCase)
  foreach ($field in $FieldPaths) {
    if (-not [string]::IsNullOrWhiteSpace($field)) {
      [void]$fieldSet.Add($field)
    }
  }

  $missingCanonical = @($RequiredCanonicalFields | Where-Object {
      $canonical = $_
      -not ($FieldPaths | Where-Object { $_ -eq $canonical -or $_.EndsWith(".$canonical", [StringComparison]::OrdinalIgnoreCase) -or $_.Contains($canonical, [StringComparison]::OrdinalIgnoreCase) })
    })

  [pscustomobject]@{
    id = $Id
    title = $Title
    artifactPath = $ArtifactPath
    recordKind = $RecordKind
    state = $State
    fieldPathCount = $FieldPaths.Count
    fieldPaths = @($FieldPaths)
    aliases = @($Aliases)
    requiredCanonicalFields = @($RequiredCanonicalFields)
    missingCanonicalFields = @($missingCanonical)
    missingCanonicalFieldCount = @($missingCanonical).Count
    ready = $false
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner input contract surface only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

function Get-OwnerExternalResultFieldPaths {
  param([AllowNull()][object]$Record)

  $fields = New-Object System.Collections.Generic.List[string]
  $resultInputs = @(Get-PropertyOrDefault -Object $Record -Name "resultInputs" -DefaultValue @())
  foreach ($input in $resultInputs) {
    $fields.AddRange([string[]](ConvertTo-StringArray (Get-PropertyOrDefault -Object $input -Name "requiredResultFields" -DefaultValue @())))
    foreach ($name in @("stdoutPath", "stderrPath", "mergedTranscriptPath", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "validatorOutputPath", "validatorOutputSha256", "ownerReviewer", "ownerReviewTimestampUtc", "nonSubstituteConfirmations", "passed")) {
      if ($input.PSObject.Properties.Name -contains $name) { $fields.Add($name) | Out-Null }
    }
  }
  return @($fields | Select-Object -Unique)
}

function Get-PublicPublishContractFieldPaths {
  param([AllowNull()][object]$Record)

  $fields = New-Object System.Collections.Generic.List[string]
  foreach ($field in @(Get-PropertyOrDefault -Object $Record -Name "requiredFields" -DefaultValue @())) {
    $name = [string](Get-PropertyOrDefault -Object $field -Name "name" -DefaultValue "")
    if (-not [string]::IsNullOrWhiteSpace($name)) { $fields.Add($name) | Out-Null }
  }
  return @($fields | Select-Object -Unique)
}

function Get-PublicPackageFieldPaths {
  param([AllowNull()][object]$Record)
  return @(ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "requiredRealInputFields" -DefaultValue @()) | Select-Object -Unique)
}

function Get-PostPublishFieldPaths {
  param([AllowNull()][object]$Record)

  $fields = @(
    "selectedChannel",
    "channelSourceUri",
    "publishedPackageUrl",
    "packageIdentity.managedPackageUrl",
    "packageIdentity.runtimePackageUrl",
    "packageIdentity.managedNupkgSha256",
    "packageIdentity.runtimeNupkgSha256",
    "packageIdentity.managedPackageSha256Source",
    "packageIdentity.runtimePackageSha256Source",
    "restoreLogPath",
    "restoreLogSha256",
    "nativeAssetListingPath",
    "nativeAssetListingSha256",
    "dependencyProbeLogPath",
    "dependencyProbeLogSha256",
    "smokeLogPath",
    "smokeLogSha256",
    "stdoutSummary",
    "stderrSummary",
    "managedPackageSource",
    "runtimePackageSource",
    "host.ownerName",
    "host.driverVersion",
    "host.cudaRuntimeVersion",
    "host.tensorRtRuntimeVersion"
  )

  return @($fields)
}

function Get-ReleaseCloseDecisionFieldPaths {
  param([AllowNull()][object]$Record)
  return @(ConvertTo-StringArray (Get-PropertyOrDefault -Object $Record -Name "requiredOwnerFields" -DefaultValue @()) | Select-Object -Unique)
}

$ownerExternalResultTemplate = Read-JsonOrNull "artifacts\final-release\owner-external-proof-execution-result.input.template.json"
$publicPublishContract = Read-JsonOrNull "artifacts\final-release\public-publish-real-result-owner-input-contract.json"
$publicPackageInput = Read-JsonOrNull "artifacts\final-release\public-package-proof-owner-input.template.json"
$postPublishInput = Read-JsonOrNull "artifacts\final-release\post-publish-verification-owner-input.template.json"
$releaseCloseOwnerDecision = Read-JsonOrNull "artifacts\final-release\release-issue-close-owner-decision-input.template.json"
$cleanExternalRunbook = Read-JsonOrNull "artifacts\final-release\clean-external-package-consumer-owner-runbook.json"
$postPublishRunbook = Read-JsonOrNull "artifacts\final-release\post-publish-owner-verification-runbook.json"

$canonicalFields = @(
  [pscustomobject]@{ name = "publicPackageUrl"; description = "Stable public package URL observed after owner publish."; aliases = @("publishedPackageUrl", "managedPackage.packageUrl", "runtimePackage.packageUrl", "packageIdentity.managedPackageUrl", "packageIdentity.runtimePackageUrl", "nugetPackageUrl") },
  [pscustomobject]@{ name = "publicPackageSourceUrl"; description = "NuGet/GitHub Packages/GitHub Release source URL used by owner and clean consumer."; aliases = @("publicSource", "publicSourceUrl", "channelSourceUri", "managedPackage.publicSourceUrl", "runtimePackage.publicSourceUrl", "managedPackageSource", "runtimePackageSource", "packageIdentity.packageSource") },
  [pscustomobject]@{ name = "downloadedNupkgSha256"; description = "SHA256 of the package downloaded from the selected public source."; aliases = @("publicPackageSha256", "managedPackage.nupkgSha256", "runtimePackage.nupkgSha256", "packageIdentity.managedNupkgSha256", "packageIdentity.runtimeNupkgSha256", "packageIdentity.nupkgSha256") },
  [pscustomobject]@{ name = "publishedTimestampUtc"; description = "UTC timestamp for the observed public package publish or download."; aliases = @("publishedAtUtc", "publishTimestampUtc", "packageIdentity.managedPackageDownloadTimestampUtc", "packageIdentity.runtimePackageDownloadTimestampUtc") },
  [pscustomobject]@{ name = "stdoutPath"; description = "Path to owner-captured stdout log."; aliases = @("stdoutPath", "restoreLogPath", "smokeLogPath") },
  [pscustomobject]@{ name = "stderrPath"; description = "Path to owner-captured stderr log."; aliases = @("stderrPath") },
  [pscustomobject]@{ name = "mergedTranscriptPath"; description = "Path to owner-captured merged command transcript."; aliases = @("mergedTranscriptPath", "publishCommandTranscriptPath") },
  [pscustomobject]@{ name = "stdoutSha256"; description = "SHA256 for stdout log."; aliases = @("stdoutSha256", "restoreLogSha256", "smokeLogSha256") },
  [pscustomobject]@{ name = "stderrSha256"; description = "SHA256 for stderr log."; aliases = @("stderrSha256") },
  [pscustomobject]@{ name = "mergedTranscriptSha256"; description = "SHA256 for merged command transcript."; aliases = @("mergedTranscriptSha256", "publishCommandTranscriptSha256") },
  [pscustomobject]@{ name = "hostMetadata"; description = "Owner host OS/GPU/driver/CUDA/TensorRT/.NET metadata."; aliases = @("hostMetadata", "host", "host.driverVersion", "host.cudaRuntimeVersion", "host.tensorRtRuntimeVersion") },
  [pscustomobject]@{ name = "ownerReviewer"; description = "Human owner/reviewer who checked real evidence."; aliases = @("ownerReviewer", "reviewer", "reviewerName", "ownerName") },
  [pscustomobject]@{ name = "ownerReviewTimestampUtc"; description = "UTC timestamp of owner review."; aliases = @("ownerReviewTimestampUtc", "reviewedAtUtc") },
  [pscustomobject]@{ name = "nonSubstituteConfirmations"; description = "Owner confirmations that no local feed, ProjectReference, direct nupkg, template, dry-run, dashboard, or candidate substituted proof."; aliases = @("nonSubstituteConfirmations", "ownerConfirmation") }
)

$publicProofCanonical = @("publicPackageUrl", "publicPackageSourceUrl", "downloadedNupkgSha256", "publishedTimestampUtc", "ownerReviewer", "ownerReviewTimestampUtc")
$runtimeProofCanonical = @("stdoutPath", "stderrPath", "mergedTranscriptPath", "stdoutSha256", "stderrSha256", "mergedTranscriptSha256", "hostMetadata", "ownerReviewer", "ownerReviewTimestampUtc", "nonSubstituteConfirmations")

$surfaces = @(
  New-ContractSurface -Id "owner-external-proof-execution-result-input" -Title "Owner external proof execution result input" -ArtifactPath "artifacts/final-release/owner-external-proof-execution-result.input.template.json" -RecordKind ([string](Get-PropertyOrDefault -Object $ownerExternalResultTemplate -Name "recordKind" -DefaultValue "missing")) -State ([string](Get-PropertyOrDefault -Object $ownerExternalResultTemplate -Name "templateState" -DefaultValue "missing-owner-external-proof-execution-result-input-template")) -FieldPaths (Get-OwnerExternalResultFieldPaths -Record $ownerExternalResultTemplate) -Aliases @("resultInputs[]") -RequiredCanonicalFields $runtimeProofCanonical
  New-ContractSurface -Id "public-publish-real-result-owner-input-contract" -Title "Public publish real result owner input contract" -ArtifactPath "artifacts/final-release/public-publish-real-result-owner-input-contract.json" -RecordKind ([string](Get-PropertyOrDefault -Object $publicPublishContract -Name "recordKind" -DefaultValue "missing")) -State ([string](Get-PropertyOrDefault -Object $publicPublishContract -Name "contractState" -DefaultValue "missing-public-publish-real-result-owner-input-contract")) -FieldPaths (Get-PublicPublishContractFieldPaths -Record $publicPublishContract) -Aliases @("publicSource", "publicPackageSha256") -RequiredCanonicalFields $publicProofCanonical
  New-ContractSurface -Id "public-package-proof-owner-input" -Title "Public package proof owner input" -ArtifactPath "artifacts/final-release/public-package-proof-owner-input.template.json" -RecordKind ([string](Get-PropertyOrDefault -Object $publicPackageInput -Name "recordKind" -DefaultValue "missing")) -State ([string](Get-PropertyOrDefault -Object $publicPackageInput -Name "templateState" -DefaultValue "missing-public-package-proof-owner-input")) -FieldPaths (Get-PublicPackageFieldPaths -Record $publicPackageInput) -Aliases @("managedPackage.publicSourceUrl", "runtimePackage.publicSourceUrl") -RequiredCanonicalFields $publicProofCanonical
  New-ContractSurface -Id "post-publish-verification-owner-input" -Title "Post-publish verification owner input" -ArtifactPath "artifacts/final-release/post-publish-verification-owner-input.template.json" -RecordKind ([string](Get-PropertyOrDefault -Object $postPublishInput -Name "recordKind" -DefaultValue "missing")) -State ([string](Get-PropertyOrDefault -Object $postPublishInput -Name "ownerInputState" -DefaultValue "missing-post-publish-verification-owner-input")) -FieldPaths (Get-PostPublishFieldPaths -Record $postPublishInput) -Aliases @("channelSourceUri", "packageIdentity.managedNupkgSha256", "packageIdentity.runtimeNupkgSha256") -RequiredCanonicalFields @($publicProofCanonical + @("stdoutPath", "stdoutSha256", "hostMetadata"))
  New-ContractSurface -Id "release-issue-close-owner-decision-input" -Title "Release issue close owner decision input" -ArtifactPath "artifacts/final-release/release-issue-close-owner-decision-input.template.json" -RecordKind ([string](Get-PropertyOrDefault -Object $releaseCloseOwnerDecision -Name "recordKind" -DefaultValue "missing")) -State ([string](Get-PropertyOrDefault -Object $releaseCloseOwnerDecision -Name "templateState" -DefaultValue "missing-release-issue-close-owner-decision-input")) -FieldPaths (Get-ReleaseCloseDecisionFieldPaths -Record $releaseCloseOwnerDecision) -Aliases @("ownerFinalCloseDecision") -RequiredCanonicalFields @("ownerReviewer", "ownerReviewTimestampUtc")
)

$runbookInputs = @(
  [pscustomobject]@{
    id = "clean-external-package-consumer-owner-runbook"
    artifactPath = "artifacts/final-release/clean-external-package-consumer-owner-runbook.json"
    state = [string](Get-PropertyOrDefault -Object $cleanExternalRunbook -Name "runbookState" -DefaultValue "missing-clean-external-package-consumer-owner-runbook")
    stepCount = [int](Get-PropertyOrDefault -Object $cleanExternalRunbook -Name "stepCount" -DefaultValue 0)
    requiredInputTarget = [string](Get-PropertyOrDefault -Object $cleanExternalRunbook -Name "requiredInputTarget" -DefaultValue "")
    fillableTemplate = [string](Get-PropertyOrDefault -Object $cleanExternalRunbook -Name "fillableTemplate" -DefaultValue "")
    boundary = "Runbook guidance only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  },
  [pscustomobject]@{
    id = "post-publish-owner-verification-runbook"
    artifactPath = "artifacts/final-release/post-publish-owner-verification-runbook.json"
    state = [string](Get-PropertyOrDefault -Object $postPublishRunbook -Name "runbookState" -DefaultValue "missing-post-publish-owner-verification-runbook")
    stepCount = [int](Get-PropertyOrDefault -Object $postPublishRunbook -Name "stepCount" -DefaultValue 0)
    requiredInputTarget = [string](Get-PropertyOrDefault -Object $postPublishRunbook -Name "requiredInputTarget" -DefaultValue "")
    fillableTemplate = [string](Get-PropertyOrDefault -Object $postPublishRunbook -Name "fillableTemplate" -DefaultValue "")
    boundary = "Runbook guidance only; does not run dotnet nuget push; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
)

$fieldCoverage = foreach ($canonical in $canonicalFields) {
  $aliases = @($canonical.aliases + $canonical.name)
  $surfaceMatches = foreach ($surface in $surfaces) {
    $matched = @($surface.fieldPaths | Where-Object {
        $field = [string]$_
        @($aliases | Where-Object { $field -eq $_ -or $field.EndsWith(".$_", [StringComparison]::OrdinalIgnoreCase) -or $field.Contains($_, [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0
      })
    [pscustomobject]@{
      surfaceId = $surface.id
      matchedFieldPaths = @($matched)
      matched = @($matched).Count -gt 0
    }
  }
  [pscustomobject]@{
    canonicalName = $canonical.name
    description = $canonical.description
    aliases = @($canonical.aliases)
    coveredSurfaceCount = @($surfaceMatches | Where-Object { $_.matched }).Count
    surfaceMatches = @($surfaceMatches)
    ready = $false
    boundary = "Canonical owner input field mapping only; not proof and not close approval."
  }
}

$blockedReasons = @(
  "real owner input files are not imported",
  "external clean consumer logs and hashes are not validated",
  "public package source URL and downloaded nupkg SHA256 are still owner-fill fields",
  "post-publish clean consumer proof is not promoted",
  "strict close record cannot close until strict validators accept real proof"
)

$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct nupkg",
  "direct .nupkg",
  "template",
  "draft",
  "candidate",
  "dashboard",
  "dry-run",
  "build-only",
  "runbook as proof",
  "manual handoff as proof"
)

$record = [ordered]@{
  recordKind = "owner-input-contract-convergence"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  convergenceState = "blocked-owner-input-contract-convergence-real-owner-input-required"
  contractSurfaceCount = $surfaces.Count
  canonicalFieldCount = $canonicalFields.Count
  runbookInputCount = $runbookInputs.Count
  blockedContractSurfaceCount = @($surfaces | Where-Object { -not [bool]$_.ready }).Count
  readyContractSurfaceCount = 0
  canonicalFields = @($canonicalFields)
  fieldCoverage = @($fieldCoverage)
  contractSurfaces = @($surfaces)
  runbookInputs = @($runbookInputs)
  blockedReasons = @($blockedReasons)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  sourceArtifacts = @(
    "artifacts/final-release/owner-external-proof-execution-result.input.template.json",
    "artifacts/final-release/public-publish-real-result-owner-input-contract.json",
    "artifacts/final-release/public-package-proof-owner-input.template.json",
    "artifacts/final-release/post-publish-verification-owner-input.template.json",
    "artifacts/final-release/release-issue-close-owner-decision-input.template.json",
    "artifacts/final-release/clean-external-package-consumer-owner-runbook.json",
    "artifacts/final-release/post-publish-owner-verification-runbook.json"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Owner input contract convergence is a schema and terminology audit only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "owner-input-contract-convergence.json"
$markdownPath = Join-Path $OutputRoot "owner-input-contract-convergence.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 18)
$fieldRows = foreach ($field in $fieldCoverage) {
  "| $(ConvertTo-MarkdownCell $field.canonicalName) | ``$($field.coveredSurfaceCount)`` | $(ConvertTo-MarkdownCell (($field.aliases -join ', '))) |"
}

$surfaceRows = foreach ($surface in $surfaces) {
  "| $(ConvertTo-MarkdownCell $surface.id) | $(ConvertTo-MarkdownCell $surface.state) | ``$($surface.fieldPathCount)`` | ``$($surface.missingCanonicalFieldCount)`` |"
}

$runbookRows = foreach ($runbook in $runbookInputs) {
  "| $(ConvertTo-MarkdownCell $runbook.id) | $(ConvertTo-MarkdownCell $runbook.state) | ``$($runbook.stepCount)`` | $(ConvertTo-MarkdownCell $runbook.requiredInputTarget) |"
}

$markdown = @"
# Owner Input Contract Convergence

| 项目 | 当前值 |
|---|---|
| convergenceState | ``$($record.convergenceState)`` |
| contractSurfaceCount | ``$($record.contractSurfaceCount)`` |
| canonicalFieldCount | ``$($record.canonicalFieldCount)`` |
| runbookInputCount | ``$($record.runbookInputCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Canonical Fields

| Canonical Field | Covered Surfaces | Aliases |
|---|---:|---|
$($fieldRows -join "`r`n")

## Contract Surfaces

| Surface | State | Field Paths | Missing Canonical |
|---|---|---:|---:|
$($surfaceRows -join "`r`n")

## Runbook Inputs

| Runbook | State | Steps | Input Target |
|---|---|---:|---|
$($runbookRows -join "`r`n")

## Boundary

$($record.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner input contract convergence written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ConvergenceState=$($record.convergenceState) Surfaces=$($record.contractSurfaceCount) CanonicalFields=$($record.canonicalFieldCount) Runbooks=$($record.runbookInputCount)"
