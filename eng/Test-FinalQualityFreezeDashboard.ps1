[CmdletBinding()]
param(
  [string]$DashboardPath,
  [string]$OutputDirectory,
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}

if ([string]::IsNullOrWhiteSpace($DashboardPath)) {
  $DashboardPath = Join-Path $OutputDirectory "final-quality-freeze-dashboard.json"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8FileWithRetry {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject,
    [int]$MaxAttempts = 8,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $content = @($InputObject) -join [Environment]::NewLine
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($LiteralPath)), [Guid]::NewGuid().ToString("N"))
  [IO.File]::WriteAllText($tempPath, $content + [Environment]::NewLine, $script:utf8)

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Move-Item -LiteralPath $tempPath -Destination $LiteralPath -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
  }
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function Add-Finding {
  param(
    [System.Collections.Generic.List[object]]$Findings,
    [string]$Id,
    [string]$Message
  )

  $Findings.Add([pscustomobject]@{
      id = $Id
      severity = "blocker"
      message = $Message
    })
}

if (-not (Test-Path -LiteralPath $DashboardPath -PathType Leaf)) {
  throw "Final quality freeze dashboard not found: $DashboardPath"
}

$dashboard = Get-Content -LiteralPath $DashboardPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = New-Object System.Collections.Generic.List[object]

if ([string](Get-PropertyOrDefault -Object $dashboard -Name "recordKind" -DefaultValue "") -ne "final-quality-freeze-dashboard") {
  Add-Finding $findings "record-kind" "recordKind must be final-quality-freeze-dashboard."
}

if ([string](Get-PropertyOrDefault -Object $dashboard -Name "freezeState" -DefaultValue "") -ne "blocked-final-quality-freeze-real-proof-required") {
  Add-Finding $findings "freeze-state" "freezeState must remain blocked-final-quality-freeze-real-proof-required."
}

foreach ($flag in @("canPublishPublicly", "canExecutePublicPublish", "canCloseReleaseIssue", "canPromoteRuntimeProof", "performsPublish", "isRuntimeExecutionProof", "isPostPublishProof", "isReleaseCloseProof", "isPackagePush")) {
  if ([bool](Get-PropertyOrDefault -Object $dashboard -Name $flag -DefaultValue $true)) {
    Add-Finding $findings "flag-$flag" "$flag must remain false."
  }
}

if ([int](Get-PropertyOrDefault -Object $dashboard -Name "failedBlockerCount" -DefaultValue -1) -ne 0) {
  Add-Finding $findings "failed-blocker-count" "failedBlockerCount must remain 0; action-required lanes carry the remaining work."
}

if ([int](Get-PropertyOrDefault -Object $dashboard -Name "failedActionRequiredCount" -DefaultValue 0) -le 0) {
  Add-Finding $findings "failed-action-required-count" "failedActionRequiredCount must be greater than 0 until real owner evidence is imported."
}

if ([int](Get-PropertyOrDefault -Object $dashboard -Name "inputRecordCount" -DefaultValue 0) -lt 21) {
  Add-Finding $findings "input-record-count" "Dashboard must aggregate the final owner input, strict close, cross-check, and final-release records."
}

if ([int](Get-PropertyOrDefault -Object $dashboard -Name "ownerRealInputControlRecordCount" -DefaultValue 0) -lt 14) {
  Add-Finding $findings "owner-real-input-control-count" "Dashboard must aggregate the owner real input contract/import/cross-check/strict close control records."
}

if ([int](Get-PropertyOrDefault -Object $dashboard -Name "unexpectedProofFlagCount" -DefaultValue 999) -ne 0) {
  Add-Finding $findings "unexpected-proof-flags" "Input records must not expose proof/publish/close flags as true."
}

$boundary = [string](Get-PropertyOrDefault -Object $dashboard -Name "boundary" -DefaultValue "")
foreach ($marker in @("not runtime proof", "not post-publish proof", "not publish approval", "not release close approval", "not package push", "failedBlockerCount=0 is not ready")) {
  if ($boundary.IndexOf($marker, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
    Add-Finding $findings "missing-boundary-marker" "Boundary is missing marker: $marker"
  }
}

$inputRecords = @((Get-PropertyOrDefault -Object $dashboard -Name "inputRecords" -DefaultValue @()))
$inputIds = @($inputRecords | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
foreach ($requiredInputId in @(
    "release-candidate-real-proof-final-freeze",
    "owner-real-input-import-preflight",
    "public-package-hash-cross-check-gate",
    "clean-consumer-runtime-proof-cross-check-gate",
    "post-publish-rollback-owner-decision-gate",
    "release-close-final-real-input-admission-pack",
    "owner-real-input-json-contract",
    "owner-real-input-json-import",
    "owner-real-input-hash-and-path-validator",
    "owner-real-input-forbidden-substitute-validator",
    "strict-close-real-input-dry-run",
    "strict-close-real-input-finding-report",
    "strict-close-owner-action-pack",
    "release-close-real-input-final-blocker-ledger"
  )) {
  if ($inputIds -notcontains $requiredInputId) {
    Add-Finding $findings "missing-input-$requiredInputId" "Dashboard is missing required owner real input control record: $requiredInputId"
  }
}

$validationState = if ($findings.Count -eq 0) { "blocked-final-quality-freeze-real-proof-required" } else { "final-quality-freeze-dashboard-validation-failed" }
$validation = [pscustomobject]@{
  recordKind = "final-quality-freeze-dashboard-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  dashboardPath = $DashboardPath
  findingCount = $findings.Count
  failedBlockerCount = 0
  failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $dashboard -Name "failedActionRequiredCount" -DefaultValue 0)
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  performsPublish = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  findings = @($findings.ToArray())
  boundary = "This validation confirms the dashboard remains blocked/non-proof only: not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push. failedBlockerCount=0 is not ready."
}

$jsonPath = Join-Path $OutputDirectory "final-quality-freeze-dashboard-validation.json"
$markdownPath = Join-Path $OutputDirectory "final-quality-freeze-dashboard-validation.md"
$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Final Quality Freeze Dashboard Validation")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| validationState | ``$($validation.validationState)`` |")
$lines.Add("| findingCount | ``$($validation.findingCount)`` |")
$lines.Add("| failedBlockerCount | ``$($validation.failedBlockerCount)`` |")
$lines.Add("| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |")
$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
if ($findings.Count -eq 0) {
  $lines.Add("- No findings. Dashboard remains blocked/non-proof.")
}
else {
  foreach ($finding in $findings) {
    $lines.Add("- ``$($finding.id)`` $($finding.message)")
  }
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($validation.boundary)

Write-Utf8FileWithRetry -LiteralPath $markdownPath -InputObject $lines

Write-Host "Final quality freeze dashboard validation written: $jsonPath"
Write-Host "Final quality freeze dashboard validation markdown written: $markdownPath"
Write-Host "ValidationState=$validationState FindingCount=$($findings.Count)"

if ($Strict -and $findings.Count -ne 0) {
  throw "Final quality freeze dashboard validation failed with $($findings.Count) finding(s)."
}
