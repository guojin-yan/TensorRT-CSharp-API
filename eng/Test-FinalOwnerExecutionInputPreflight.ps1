[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-input-skeleton.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  $content = (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine)
  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $content, $script:utf8)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
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

function New-PreflightFinding {
  param([string]$Id, [string]$Severity, [string]$Category, [string]$FieldPath, [string]$Message)
  [pscustomobject]@{
    id = $Id
    severity = $Severity
    category = $Category
    fieldPath = $FieldPath
    message = $Message
    ownerActionRequired = $true
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final owner execution input skeleton not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$groups = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "fieldGroups" -DefaultValue @()))
$fields = @($groups | ForEach-Object { Convert-ToArray (Get-PropertyOrDefault -Object $_ -Name "fields" -DefaultValue @()) })
$findings = New-Object System.Collections.Generic.List[object]

foreach ($field in $fields) {
  $id = [string](Get-PropertyOrDefault -Object $field -Name "id" -DefaultValue "")
  $fieldPath = [string](Get-PropertyOrDefault -Object $field -Name "fieldPath" -DefaultValue "")
  $kind = [string](Get-PropertyOrDefault -Object $field -Name "kind" -DefaultValue "")
  $value = [string](Get-PropertyOrDefault -Object $field -Name "value" -DefaultValue "")
  $isPlaceholder = [bool](Get-PropertyOrDefault -Object $field -Name "placeholder" -DefaultValue $false)

  if ($isPlaceholder -or [string]::IsNullOrWhiteSpace($value) -or $value -eq "<owner-real-input-required>") {
    $findings.Add((New-PreflightFinding -Id "$id-placeholder" -Severity "action-required" -Category "placeholder" -FieldPath $fieldPath -Message "Owner must replace placeholder with real input.")) | Out-Null
  }

  if ($kind -eq "sha256" -and -not [System.Text.RegularExpressions.Regex]::IsMatch($value, "^[0-9a-fA-F]{64}$")) {
    $findings.Add((New-PreflightFinding -Id "$id-sha256-invalid" -Severity "action-required" -Category "SHA256 invalid" -FieldPath $fieldPath -Message "SHA256 must be 64 hexadecimal characters.")) | Out-Null
  }

  if ($kind -eq "path") {
    if ($RequireExistingFiles.IsPresent) {
      $path = Resolve-RepositoryPath -Path $value
      if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $findings.Add((New-PreflightFinding -Id "$id-path-missing" -Severity "action-required" -Category "path missing" -FieldPath $fieldPath -Message "Path must exist when -RequireExistingFiles is specified.")) | Out-Null
      }
    }
    elseif ($isPlaceholder) {
      $findings.Add((New-PreflightFinding -Id "$id-path-missing" -Severity "action-required" -Category "path missing" -FieldPath $fieldPath -Message "Path is still placeholder and cannot be imported.")) | Out-Null
    }
  }
}

$raw = $record | ConvertTo-Json -Depth 20
foreach ($forbidden in @("local feed", "ProjectReference", "direct .nupkg", "pre-publish smoke reused as post-publish proof")) {
  $findings.Add((New-PreflightFinding -Id "forbidden-substitute-$($forbidden.Replace(' ', '-').Replace('.', 'dot'))" -Severity "action-required" -Category "forbidden substitute rejected" -FieldPath "nonSubstituteConfirmations" -Message "Preflight rejects substitute marker: $forbidden")) | Out-Null
}

$findings.Add((New-PreflightFinding -Id "strict-validator-not-run" -Severity "action-required" -Category "strict validator not run" -FieldPath "strictValidators.outputPath" -Message "Strict validator output is required before import.")) | Out-Null
if ($RequireHashMatch.IsPresent) {
  $findings.Add((New-PreflightFinding -Id "hash-match-not-available" -Severity "action-required" -Category "SHA256 invalid" -FieldPath "hashMatch" -Message "Hash matching cannot pass until Owner provides real files and SHA256 values.")) | Out-Null
}

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$readyForImport = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$preflightState = if ($readyForImport) { "ready-for-owner-input-import-candidate" } else { "blocked-final-owner-real-input-required" }

$preflight = [pscustomobject]@{
  recordKind = "final-owner-execution-input-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  preflightState = $preflightState
  inputPath = $resolvedInputPath
  strict = $Strict.IsPresent
  requireExistingFiles = $RequireExistingFiles.IsPresent
  requireHashMatch = $RequireHashMatch.IsPresent
  checkedFieldCount = $fields.Count
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  readyForImport = $readyForImport
  ownerActionRequired = -not $readyForImport
  findings = @($findings.ToArray())
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  nonSubstituteProofKinds = @("final owner execution input preflight", "strict owner input preflight", "forbidden substitute rejection")
  sourceArtifacts = @("artifacts/final-release/final-owner-execution-input-skeleton.json")
  boundary = "Final Owner execution input preflight validates placeholder, path, SHA256, strict validator, and forbidden substitute conditions only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-input-preflight.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-input-preflight.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($preflight | ConvertTo-Json -Depth 14)
$rows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | ``$(ConvertTo-MarkdownCell $finding.fieldPath)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}

$markdown = @"
# Final Owner Execution Input Preflight

| Field | Value |
|---|---|
| preflightState | ``$($preflight.preflightState)`` |
| checkedFieldCount | ``$($preflight.checkedFieldCount)`` |
| findingCount | ``$($preflight.findingCount)`` |
| failedBlockerCount | ``$($preflight.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($preflight.failedActionRequiredCount)`` |
| readyForImport | ``$($preflight.readyForImport)`` |
| requireExistingFiles | ``$($preflight.requireExistingFiles)`` |
| requireHashMatch | ``$($preflight.requireHashMatch)`` |

## Findings

| ID | Category | Field | Message |
|---|---|---|---|
$($rows -join "`r`n")

## Boundary

$($preflight.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution input preflight written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "PreflightState=$preflightState ActionRequired=$($failedActionRequired.Count) Ready=$readyForImport"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution input preflight failed with $($failedBlockers.Count) blocker(s)."
}
