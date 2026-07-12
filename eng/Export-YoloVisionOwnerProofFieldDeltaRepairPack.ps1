[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json",
  [string]$ValidationPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-input-validation.json",
  [string]$ExecutionPackPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-execution-pack.json",
  [string]$ImportReportPath = "artifacts/user-acceptance/yolovision-real-asset-owner-proof-import-report.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

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

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
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

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-FieldDelta {
  param(
    [string]$Id,
    [string]$CaseId,
    [string]$JsonPath,
    [string]$Category,
    [string]$Description,
    [string]$Example,
    [string]$ValidatorItemId,
    [string]$RepairOrder
  )

  [pscustomobject]@{
    id = $Id
    caseId = $CaseId
    jsonPath = $JsonPath
    category = $Category
    description = $Description
    example = $Example
    validatorItemId = $ValidatorItemId
    repairOrder = $RepairOrder
    ownerActionRequired = $true
    canAutoFill = $false
    canPromoteProof = $false
  }
}

function Get-CategoryFromValidationId {
  param([string]$Id)

  if ($Id -like "global-*") { return "global-host-metadata" }
  if ($Id -like "*sha256*" -or $Id -like "*hash*") { return "hash" }
  if ($Id -like "*path*" -or $Id -like "*log*" -or $Id -like "*json*") { return "path-or-log" }
  if ($Id -like "*owner*" -or $Id -like "*review*") { return "owner-review" }
  if ($Id -like "*expected*" -or $Id -like "*passed*") { return "expected-evidence-line" }
  return "owner-input"
}

function Get-JsonPathFromValidationId {
  param([string]$Id)

  if ($Id -like "global-*") {
    return "requiredGlobalEvidence.$($Id.Substring(7))"
  }

  if ($Id -match "^case-([^-]+-[^-]+)-(.+)$") {
    $caseId = $Matches[1]
    $field = $Matches[2]
    $fieldPath = $field.Replace("-", ".")
    return "cases[$caseId].$fieldPath"
  }

  return "validationItems[$Id]"
}

function Get-ExampleFromCategory {
  param([string]$Category)

  switch ($Category) {
    "global-host-metadata" { return "Windows 11 x64, RTX 4090, driver/CUDA/TensorRT/runtime package version, owner and reviewedAtUtc." }
    "hash" { return "64-character lowercase SHA256 from Get-FileHash -Algorithm SHA256." }
    "path-or-log" { return "Repository-relative real artifact path plus matching SHA256 and existing log/output JSON file." }
    "owner-review" { return "Named owner reviewer, reviewedAtUtc in ISO-8601, accepted flag and notes." }
    "expected-evidence-line" { return "Owner-provided real run log line matching the expected YoloVision success marker." }
    default { return "Replace owner-required placeholder with real owner-provided evidence." }
  }
}

function Get-RepairOrderFromCategory {
  param([string]$Category)

  switch ($Category) {
    "global-host-metadata" { return "01-host-metadata" }
    "path-or-log" { return "02-real-artifact-paths" }
    "hash" { return "03-sha256-hashes" }
    "expected-evidence-line" { return "04-run-log-evidence" }
    "owner-review" { return "05-owner-review" }
    default { return "06-other-owner-input" }
  }
}

$ownerInput = Read-JsonOrNull $OwnerInputPath
$validation = Read-JsonOrNull $ValidationPath
$executionPack = Read-JsonOrNull $ExecutionPackPath
$importReport = Read-JsonOrNull $ImportReportPath

if ($null -eq $ownerInput) {
  throw "Owner input was not found: $OwnerInputPath"
}

$validationItems = @()
if ($null -ne $validation) {
  $validationItems = @(Get-PropertyOrDefault -Object $validation -Name "validationItems" -DefaultValue @())
}

$failedOwnerItems = @(
  $validationItems |
    Where-Object {
      -not [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false) -and
      [string](Get-PropertyOrDefault -Object $_ -Name "severity" -DefaultValue "") -in @("owner-action-required", "action-required")
    }
)

$fieldDeltas = @(
  foreach ($item in $failedOwnerItems) {
    $id = [string](Get-PropertyOrDefault -Object $item -Name "id" -DefaultValue "")
    $detail = [string](Get-PropertyOrDefault -Object $item -Name "detail" -DefaultValue "")
    $caseId = "global"
    if ($id -match "^case-([^-]+-[^-]+)-") {
      $caseId = $Matches[1]
    }

    $category = Get-CategoryFromValidationId -Id $id
    New-FieldDelta `
      -Id "repair-$id" `
      -CaseId $caseId `
      -JsonPath (Get-JsonPathFromValidationId -Id $id) `
      -Category $category `
      -Description $detail `
      -Example (Get-ExampleFromCategory -Category $category) `
      -ValidatorItemId $id `
      -RepairOrder (Get-RepairOrderFromCategory -Category $category)
  }
)

$caseSummaries = @(
  @($ownerInput.cases) | ForEach-Object {
    $caseId = [string](Get-PropertyOrDefault -Object $_ -Name "caseId" -DefaultValue "unknown")
    $caseDeltas = @($fieldDeltas | Where-Object { $_.caseId -eq $caseId })
    [pscustomobject]@{
      caseId = $caseId
      task = [string](Get-PropertyOrDefault -Object $_ -Name "task" -DefaultValue "")
      missingFieldCount = $caseDeltas.Count
      hashFieldCount = @($caseDeltas | Where-Object { $_.category -eq "hash" }).Count
      pathOrLogFieldCount = @($caseDeltas | Where-Object { $_.category -eq "path-or-log" }).Count
      ownerReviewFieldCount = @($caseDeltas | Where-Object { $_.category -eq "owner-review" }).Count
      nextRepairOrder = @($caseDeltas | Sort-Object repairOrder | Select-Object -ExpandProperty repairOrder -First 1)
    }
  }
)

$globalDeltas = @($fieldDeltas | Where-Object { $_.caseId -eq "global" })
$repairPack = [ordered]@{
  recordKind = "yolovision-owner-proof-field-delta-repair-pack"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  repairPackState = "blocked-owner-action-required"
  sourceOwnerInput = $OwnerInputPath
  sourceValidation = $ValidationPath
  sourceExecutionPack = $ExecutionPackPath
  sourceImportReport = $ImportReportPath
  validationState = [string](Get-PropertyOrDefault -Object $validation -Name "validationState" -DefaultValue "missing-validation")
  importValidationState = [string](Get-PropertyOrDefault -Object $importReport -Name "validationState" -DefaultValue "missing-import-report")
  performsPublish = $false
  canAutoFillOwnerFields = $false
  canPromoteRealModelRuntime = $false
  canPromotePackageConsumerRuntime = $false
  canCloseReleaseIssue = $false
  caseCount = @($ownerInput.cases).Count
  missingFieldCount = $fieldDeltas.Count
  globalMissingFieldCount = $globalDeltas.Count
  categoryCatalog = @(
    [pscustomobject]@{ category = "global-host-metadata"; description = "Host OS, GPU, driver, CUDA, TensorRT, runtime package, owner reviewer, and owner review time." }
    [pscustomobject]@{ category = "path-or-log"; description = "Repository-relative real artifact paths, run logs, output JSON, and any matching file existence checks." }
    [pscustomobject]@{ category = "hash"; description = "64-character SHA256 values for model, labels, input, tensor, engine, report, run log, and output JSON." }
    [pscustomobject]@{ category = "expected-evidence-line"; description = "Owner-provided real run log markers required before real-model-runtime promotion." }
    [pscustomobject]@{ category = "owner-review"; description = "Named owner review, accepted flag, review timestamp, and notes." }
    [pscustomobject]@{ category = "owner-input"; description = "Other owner-provided non-placeholder evidence fields." }
  )
  caseSummaries = @($caseSummaries)
  fieldDeltas = @($fieldDeltas)
  recommendedRepairOrder = @(
    "01-host-metadata",
    "02-real-artifact-paths",
    "03-sha256-hashes",
    "04-run-log-evidence",
    "05-owner-review",
    "06-other-owner-input",
    "07-run-Test-YoloVisionRealAssetOwnerProofInput-Strict",
    "08-run-Import-YoloVisionRealAssetOwnerProofInput",
    "09-run-Test-SampleRunEvidenceRecord-RequireExistingLog"
  )
  ownerCommands = @(
    "Fill artifacts/user-acceptance/yolovision-real-asset-owner-proof-input.template.json with real owner evidence.",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-YoloVisionRealAssetOwnerProofInput.ps1 -Strict",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-YoloVisionRealAssetOwnerProofInput.ps1",
    "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
  )
  boundary = "This repair pack is an owner-action guide only. It cannot auto-fill owner fields, cannot manufacture hashes/logs, and cannot promote real-model-runtime or package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "yolovision-owner-proof-field-delta-repair-pack.json"
$markdownPath = Join-Path $OutputRoot "yolovision-owner-proof-field-delta-repair-pack.md"
$repairPack | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$caseRows = foreach ($case in $caseSummaries) {
  "| ``$(ConvertTo-MarkdownCell $case.caseId)`` | ``$(ConvertTo-MarkdownCell $case.task)`` | $($case.missingFieldCount) | $($case.hashFieldCount) | $($case.pathOrLogFieldCount) | $($case.ownerReviewFieldCount) | ``$(ConvertTo-MarkdownCell ($case.nextRepairOrder -join ', '))`` |"
}

$deltaRows = foreach ($delta in ($fieldDeltas | Sort-Object repairOrder, caseId, validatorItemId | Select-Object -First 80)) {
  "| ``$(ConvertTo-MarkdownCell $delta.caseId)`` | ``$(ConvertTo-MarkdownCell $delta.category)`` | ``$(ConvertTo-MarkdownCell $delta.jsonPath)`` | ``$(ConvertTo-MarkdownCell $delta.validatorItemId)`` | $(ConvertTo-MarkdownCell $delta.description) |"
}

$commandLines = $repairPack.ownerCommands | ForEach-Object { "- ``$_``" }
$repairOrderLines = $repairPack.recommendedRepairOrder | ForEach-Object { "- ``$_``" }

$markdown = @"
# YoloVision Owner Proof Field Delta Repair Pack

Generated at: ``$($repairPack.generatedAtUtc)``

## Summary

- recordKind: ``$($repairPack.recordKind)``
- repairPackState: ``$($repairPack.repairPackState)``
- validationState: ``$($repairPack.validationState)``
- missingFieldCount: ``$($repairPack.missingFieldCount)``
- globalMissingFieldCount: ``$($repairPack.globalMissingFieldCount)``
- caseCount: ``$($repairPack.caseCount)``
- canAutoFillOwnerFields: ``False``
- canPromoteRealModelRuntime: ``False``
- canPromotePackageConsumerRuntime: ``False``
- canCloseReleaseIssue: ``False``

## Case Summary

| Case | Task | Missing | Hash | Path/Log | Owner Review | Next Repair |
| --- | --- | ---: | ---: | ---: | ---: | --- |
$($caseRows -join "`r`n")

## Recommended Repair Order

$($repairOrderLines -join "`r`n")

## Owner Commands

$($commandLines -join "`r`n")

## Field Deltas

| Case | Category | JSON Path | Validator Item | Detail |
| --- | --- | --- | --- | --- |
$($deltaRows -join "`r`n")

## Boundary

$($repairPack.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "YoloVision owner proof field delta repair pack written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "RepairPackState=$($repairPack.repairPackState) MissingFieldCount=$($repairPack.missingFieldCount) GlobalMissingFieldCount=$($repairPack.globalMissingFieldCount)"
