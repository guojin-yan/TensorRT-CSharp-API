[CmdletBinding()]
param(
  [string]$TemplatePath = "artifacts\final-release\final-owner-execution-real-input.template.json",
  [string]$ExamplePath = "artifacts\final-release\final-owner-execution-real-input.example.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("TemplatePath", "ExamplePath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionRealInputTemplate.ps1") -RepositoryRoot $RepositoryRoot
}

if (-not (Test-Path -LiteralPath $TemplatePath -PathType Leaf)) { throw "Template not found: $TemplatePath" }
if (-not (Test-Path -LiteralPath $ExamplePath -PathType Leaf)) { throw "Example not found: $ExamplePath" }

$template = Get-Content -LiteralPath $TemplatePath -Raw -Encoding utf8 | ConvertFrom-Json
$example = Get-Content -LiteralPath $ExamplePath -Raw -Encoding utf8 | ConvertFrom-Json
$fieldValueProperties = @($template.fieldValues.PSObject.Properties)
$fileEvidence = @(Convert-ToArray (Get-PropertyOrDefault -Object $template -Name "fileEvidence" -DefaultValue @()))
$hashEvidence = @(Convert-ToArray (Get-PropertyOrDefault -Object $template -Name "hashEvidence" -DefaultValue @()))
$confirmations = @(Convert-ToArray (Get-PropertyOrDefault -Object $template -Name "nonSubstituteConfirmations" -DefaultValue @()))
$templateText = $template | ConvertTo-Json -Depth 20
$exampleText = $example | ConvertTo-Json -Depth 20

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string]$template.recordKind -eq "final-owner-execution-real-input-template") -Severity "blocker" -Detail "Template recordKind must match.")) | Out-Null
$items.Add((New-ValidationItem -Id "required-sections" -Passed (($template.PSObject.Properties.Name -contains "owner") -and ($template.PSObject.Properties.Name -contains "fieldValues") -and ($template.PSObject.Properties.Name -contains "fileEvidence") -and ($template.PSObject.Properties.Name -contains "hashEvidence") -and ($template.PSObject.Properties.Name -contains "hostMetadata") -and ($template.PSObject.Properties.Name -contains "packageMetadata") -and ($template.PSObject.Properties.Name -contains "postPublishEvidence") -and ($template.PSObject.Properties.Name -contains "dualPackageRouteProof") -and ($template.PSObject.Properties.Name -contains "rollbackReview") -and ($template.PSObject.Properties.Name -contains "finalCloseDecision") -and ($template.PSObject.Properties.Name -contains "strictValidatorOutputs") -and ($template.PSObject.Properties.Name -contains "nonSubstituteConfirmations")) -Severity "blocker" -Detail "Template must include all Owner real input sections, including dual-package route proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "field-count" -Passed ($fieldValueProperties.Count -ge 47) -Severity "blocker" -Detail "Template must map every skeleton field, including dual-package route proof fields.")) | Out-Null
$items.Add((New-ValidationItem -Id "file-and-hash-evidence" -Passed ($fileEvidence.Count -gt 0 -and $hashEvidence.Count -gt 0) -Severity "blocker" -Detail "Template must expose file and hash evidence sections.")) | Out-Null
$items.Add((New-ValidationItem -Id "placeholder-defaults" -Passed (@($fieldValueProperties | Where-Object { [string]$_.Value -ne "<owner-real-input-required>" }).Count -eq 0) -Severity "blocker" -Detail "All template field values must default to placeholders.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-confirmations" -Passed ($confirmations.Count -ge 10 -and $templateText.Contains("local feed") -and $templateText.Contains("ProjectReference") -and $templateText.Contains("direct .nupkg") -and $templateText.Contains("pre-publish smoke reused as post-publish proof")) -Severity "blocker" -Detail "Template must carry forbidden substitute confirmations.")) | Out-Null
$items.Add((New-ValidationItem -Id "dual-package-route-proof" -Passed ($templateText.Contains("dualPackageRouteProof") -and $templateText.Contains("nuget-small-bridge-core") -and $templateText.Contains("github-packages-full-runtime") -and $templateText.Contains("owner-dual-package-nuget-owner-authorization-url") -and $templateText.Contains("owner-dual-package-github-runtime-dll-resolution-report-path") -and $templateText.Contains("Test-DualPackagePublishPreflightMatrix.ps1") -and $templateText.Contains("Test-FinalCloseGateConvergence.ps1") -and $templateText.Contains("not package push")) -Severity "blocker" -Detail "Template must expose both dual-package route proof surfaces and strict validators without promoting proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-flags" -Passed ((-not [bool]$template.performsPublish) -and (-not [bool]$template.performsRuntimeExecution) -and (-not [bool]$template.canPromoteRuntimeProof) -and (-not [bool]$template.canPublishPublicly) -and (-not [bool]$template.canCloseReleaseIssue) -and (-not [bool]$template.isRuntimeExecutionProof) -and (-not [bool]$template.isPostPublishProof) -and (-not [bool]$template.isReleaseCloseProof)) -Severity "blocker" -Detail "Template must remain non-proof and non-publish.")) | Out-Null
$items.Add((New-ValidationItem -Id "example-non-proof" -Passed ([string]$example.recordKind -eq "final-owner-execution-real-input-example" -and [bool]$example.isExample -and -not [bool]$example.readyForImport -and $exampleText.Contains("example-not-real-proof")) -Severity "blocker" -Detail "Example must be explicit non-proof and non-importable.")) | Out-Null
$items.Add((New-ValidationItem -Id "boundary" -Passed ([string]$template.boundary -match "not runtime proof" -and [string]$template.boundary -match "not post-publish proof" -and [string]$template.boundary -match "not publish approval" -and [string]$template.boundary -match "not release close approval" -and [string]$template.boundary -match "not package push") -Severity "blocker" -Detail "Boundary must exclude proof, publish, close, and package push.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "blocked-final-owner-real-input-template-ready" } else { "invalid-final-owner-execution-real-input-template" }

$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-real-input-template-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  templatePath = $TemplatePath
  examplePath = $ExamplePath
  fieldValueCount = $fieldValueProperties.Count
  fileEvidenceCount = $fileEvidence.Count
  hashEvidenceCount = $hashEvidence.Count
  nonSubstituteConfirmationCount = $confirmations.Count
  failedBlockerCount = $failedBlockers.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = $validationItems
  boundary = "Final Owner execution real input template validation checks shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-real-input-template-validation.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-real-input-template-validation.md"
$validation | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($item in $validationItems) {
  "| ``$(ConvertTo-MarkdownCell $item.id)`` | ``$($item.passed)`` | ``$(ConvertTo-MarkdownCell $item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |"
}

$markdown = @"
# Final Owner Execution Real Input Template Validation

| Field | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| fieldValueCount | ``$($validation.fieldValueCount)`` |
| fileEvidenceCount | ``$($validation.fileEvidenceCount)`` |
| hashEvidenceCount | ``$($validation.hashEvidenceCount)`` |
| nonSubstituteConfirmationCount | ``$($validation.nonSubstituteConfirmationCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution real input template validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution real input template validation failed with $($failedBlockers.Count) blocker(s)."
}
