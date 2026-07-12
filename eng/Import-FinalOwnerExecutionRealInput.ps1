[CmdletBinding()]
param(
  [string]$OwnerInputPath = "artifacts\final-release\final-owner-execution-real-input.template.json",
  [string]$SkeletonPath = "artifacts\final-release\final-owner-execution-input-skeleton.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("OwnerInputPath", "SkeletonPath", "OutputRoot")) {
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

function Resolve-InputPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Test-Placeholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("<owner-", [StringComparison]::OrdinalIgnoreCase) -or $text.Contains("example-not-real-proof", [StringComparison]::OrdinalIgnoreCase)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not (Test-Path -LiteralPath $OwnerInputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionRealInputTemplate.ps1") -RepositoryRoot $RepositoryRoot
}
if (-not (Test-Path -LiteralPath $SkeletonPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-FinalOwnerExecutionInputSkeleton.ps1") -RepositoryRoot $RepositoryRoot
}

$ownerInput = Get-Content -LiteralPath $OwnerInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$skeleton = Get-Content -LiteralPath $SkeletonPath -Raw -Encoding utf8 | ConvertFrom-Json
$groups = @(Convert-ToArray (Get-PropertyOrDefault -Object $skeleton -Name "fieldGroups" -DefaultValue @()))
$fields = @($groups | ForEach-Object { Convert-ToArray (Get-PropertyOrDefault -Object $_ -Name "fields" -DefaultValue @()) })
$fieldValues = Get-PropertyOrDefault -Object $ownerInput -Name "fieldValues" -DefaultValue ([pscustomobject]@{})
$fileEvidence = @(Convert-ToArray (Get-PropertyOrDefault -Object $ownerInput -Name "fileEvidence" -DefaultValue @()))
$hashEvidence = @(Convert-ToArray (Get-PropertyOrDefault -Object $ownerInput -Name "hashEvidence" -DefaultValue @()))
$nonSubstituteConfirmations = @(Convert-ToArray (Get-PropertyOrDefault -Object $ownerInput -Name "nonSubstituteConfirmations" -DefaultValue @()))
$forbiddenNeedles = @("local feed", "ProjectReference", "direct .nupkg", "direct nupkg", "pre-publish smoke reused as post-publish proof")

$fieldResults = New-Object System.Collections.Generic.List[object]
foreach ($field in $fields) {
  $id = [string](Get-PropertyOrDefault -Object $field -Name "id" -DefaultValue "")
  $fieldPath = [string](Get-PropertyOrDefault -Object $field -Name "fieldPath" -DefaultValue "")
  $kind = [string](Get-PropertyOrDefault -Object $field -Name "kind" -DefaultValue "")
  $value = if ($fieldValues.PSObject.Properties.Name -contains $id) { $fieldValues.PSObject.Properties[$id].Value } else { $null }
  $valueText = [string]$value
  $placeholderReplaced = -not (Test-Placeholder -Value $value)
  $sha256FormatValid = if ($kind -eq "sha256") { [System.Text.RegularExpressions.Regex]::IsMatch($valueText, "^[0-9a-fA-F]{64}$") } else { $true }
  $pathExists = $false
  $hashMatchesFile = $false
  if ($kind -eq "path" -and $placeholderReplaced) {
    $resolved = Resolve-InputPath -Path $valueText
    $pathExists = Test-Path -LiteralPath $resolved -PathType Leaf
  }
  $forbiddenFound = @($forbiddenNeedles | Where-Object { $valueText.IndexOf($_, [StringComparison]::OrdinalIgnoreCase) -ge 0 })
  $readyForImport = $placeholderReplaced -and $sha256FormatValid -and $forbiddenFound.Count -eq 0
  if ($RequireExistingFiles.IsPresent -and $kind -eq "path") { $readyForImport = $readyForImport -and $pathExists }
  if ($RequireHashMatch.IsPresent -and $kind -eq "sha256") { $readyForImport = $false }

  $fieldResults.Add([pscustomobject]@{
      fieldId = $id
      fieldPath = $fieldPath
      kind = $kind
      suppliedValue = $valueText
      placeholderReplaced = $placeholderReplaced
      pathExists = $pathExists
      sha256FormatValid = $sha256FormatValid
      hashMatchesFile = $hashMatchesFile
      forbiddenSubstituteFound = @($forbiddenFound)
      readyForImport = $readyForImport
    }) | Out-Null
}

$readyFieldCount = @($fieldResults | Where-Object { [bool]$_.readyForImport }).Count
$placeholderCount = @($fieldResults | Where-Object { -not [bool]$_.placeholderReplaced }).Count
$invalidShaCount = @($fieldResults | Where-Object { -not [bool]$_.sha256FormatValid }).Count
$missingPathCount = @($fieldResults | Where-Object { [string]$_.kind -eq "path" -and -not [bool]$_.pathExists }).Count
$forbiddenCount = @($fieldResults | Where-Object { @($_.forbiddenSubstituteFound).Count -gt 0 }).Count
$candidateReady = $readyFieldCount -eq $fieldResults.Count -and $fieldResults.Count -gt 0 -and $invalidShaCount -eq 0 -and $forbiddenCount -eq 0

$candidate = [pscustomobject]@{
  recordKind = "final-owner-execution-real-input-candidate"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  candidateState = if ($candidateReady) { "ready-for-strict-owner-input-preflight-candidate" } else { "blocked-final-owner-real-input-required" }
  ownerInputPath = $OwnerInputPath
  skeletonPath = $SkeletonPath
  fieldResultCount = $fieldResults.Count
  readyFieldCount = $readyFieldCount
  placeholderFieldCount = $placeholderCount
  invalidSha256FieldCount = $invalidShaCount
  missingPathFieldCount = $missingPathCount
  forbiddenSubstituteFieldCount = $forbiddenCount
  readyForImport = $candidateReady
  fieldResults = @($fieldResults.ToArray())
  fileEvidence = @($fileEvidence)
  hashEvidence = @($hashEvidence)
  nonSubstituteConfirmations = @($nonSubstituteConfirmations)
  ownerActionRequired = -not $candidateReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution real input candidate is an import candidate only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$import = [pscustomobject]@{
  recordKind = "final-owner-execution-real-input-import"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  importState = if ($candidateReady) { "ready-for-strict-owner-input-preflight-candidate" } else { "blocked-final-owner-real-input-required" }
  ownerInputPath = $OwnerInputPath
  candidatePath = "artifacts/final-release/final-owner-execution-real-input-candidate.json"
  overlayFieldCount = $fieldResults.Count
  readyFieldCount = $readyFieldCount
  placeholderFieldCount = $placeholderCount
  invalidSha256FieldCount = $invalidShaCount
  missingPathFieldCount = $missingPathCount
  forbiddenSubstituteFieldCount = $forbiddenCount
  requireExistingFiles = $RequireExistingFiles.IsPresent
  requireHashMatch = $RequireHashMatch.IsPresent
  nonSubstituteConfirmationCount = $nonSubstituteConfirmations.Count
  readyForImport = $candidateReady
  ownerActionRequired = -not $candidateReady
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution real input import overlays owner inputs into a candidate only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$candidateJsonPath = Join-Path $OutputRoot "final-owner-execution-real-input-candidate.json"
$candidateMarkdownPath = Join-Path $OutputRoot "final-owner-execution-real-input-candidate.md"
$importJsonPath = Join-Path $OutputRoot "final-owner-execution-real-input-import.json"
$importMarkdownPath = Join-Path $OutputRoot "final-owner-execution-real-input-import.md"
$candidate | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $candidateJsonPath -Encoding utf8
$import | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $importJsonPath -Encoding utf8

$rows = foreach ($result in $fieldResults) {
  "| ``$(ConvertTo-MarkdownCell $result.fieldId)`` | ``$(ConvertTo-MarkdownCell $result.kind)`` | ``$($result.placeholderReplaced)`` | ``$($result.sha256FormatValid)`` | ``$($result.pathExists)`` | ``$($result.readyForImport)`` |"
}

$candidateMarkdown = @"
# Final Owner Execution Real Input Candidate

| Field | Value |
|---|---|
| candidateState | ``$($candidate.candidateState)`` |
| fieldResultCount | ``$($candidate.fieldResultCount)`` |
| readyFieldCount | ``$($candidate.readyFieldCount)`` |
| placeholderFieldCount | ``$($candidate.placeholderFieldCount)`` |
| invalidSha256FieldCount | ``$($candidate.invalidSha256FieldCount)`` |
| missingPathFieldCount | ``$($candidate.missingPathFieldCount)`` |
| forbiddenSubstituteFieldCount | ``$($candidate.forbiddenSubstituteFieldCount)`` |
| readyForImport | ``$($candidate.readyForImport)`` |

## Field Results

| Field | Kind | Placeholder Replaced | SHA256 Valid | Path Exists | Ready |
|---|---|---:|---:|---:|---:|
$($rows -join "`r`n")

## Boundary

$($candidate.boundary)
"@
Write-Utf8File -LiteralPath $candidateMarkdownPath -InputObject $candidateMarkdown

$importMarkdown = @"
# Final Owner Execution Real Input Import

| Field | Value |
|---|---|
| importState | ``$($import.importState)`` |
| overlayFieldCount | ``$($import.overlayFieldCount)`` |
| readyFieldCount | ``$($import.readyFieldCount)`` |
| placeholderFieldCount | ``$($import.placeholderFieldCount)`` |
| invalidSha256FieldCount | ``$($import.invalidSha256FieldCount)`` |
| missingPathFieldCount | ``$($import.missingPathFieldCount)`` |
| forbiddenSubstituteFieldCount | ``$($import.forbiddenSubstituteFieldCount)`` |
| readyForImport | ``$($import.readyForImport)`` |

## Boundary

$($import.boundary)
"@
Write-Utf8File -LiteralPath $importMarkdownPath -InputObject $importMarkdown

Write-Host "Final owner execution real input import written:"
Write-Host "  Import=$importJsonPath"
Write-Host "  Candidate=$candidateJsonPath"
Write-Host "ImportState=$($import.importState) Ready=$($import.readyForImport) ReadyFields=$readyFieldCount/$($fieldResults.Count)"
