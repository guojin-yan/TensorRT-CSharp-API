[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$LinuxRuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

function New-ExampleDecision {
  param(
    [string]$Id,
    [string]$DecisionState,
    [string]$Rationale
  )

  [pscustomobject]@{
    id = $Id
    decisionState = $DecisionState
    ownerName = "example-owner-not-for-publication"
    decidedAtUtc = $null
    rationale = $Rationale
    evidenceUri = ""
  }
}

$decisions = @(
  New-ExampleDecision -Id "release-channel" -DecisionState "blocked" -Rationale "Example only. A real owner must choose public/private channel and rollback policy."
  New-ExampleDecision -Id "signing-policy" -DecisionState "blocked" -Rationale "Example only. A real owner must approve signed output or unsigned RC disposition."
  New-ExampleDecision -Id "nvidia-redistribution" -DecisionState "blocked" -Rationale "Example only. A real owner/legal review must decide redistribution scope."
  New-ExampleDecision -Id "runtime-proof-disposition" -DecisionState "blocked" -Rationale "Example only. Current runtime proof remains blocked-by-cuda-driver."
  New-ExampleDecision -Id "linux-runner-proof-disposition" -DecisionState "blocked" -Rationale "Example only. Linux runner proof remains template-only."
  New-ExampleDecision -Id "callback-proof-disposition" -DecisionState "blocked" -Rationale "Example only. Real callback runtime proof remains false."
)

$record = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "release-owner-approval-input-example"
  runtimePackageKey = $RuntimePackageKey
  linuxRuntimePackageKey = $LinuxRuntimePackageKey
  templateOnly = $false
  approvalState = "example-not-for-publication"
  canPublishPublicly = $false
  requestedCanPublishPublicly = $false
  ownerName = "example-owner-not-for-publication"
  ownerDecisionId = "example-not-for-publication"
  decisionInputs = $decisions
  safetyNotes = @(
    "This example is not a release-owner-approval-input-record.",
    "This example cannot approve publication.",
    "canPublishPublicly remains false.",
    "Copy the template to release-owner-approval-input-record.json only when a real owner fills accepted decision states."
  )
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "release-owner-approval-input-record.example.json"
$markdownPath = Join-Path $outputRoot "release-owner-approval-input-record.example.md"

$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Release Owner Approval Input Example")
$lines.Add("")
$lines.Add("This is an example only. It is not a real owner approval record and cannot approve publication.")
$lines.Add("")
$lines.Add("- record kind: ``release-owner-approval-input-example``")
$lines.Add("- approval state: ``example-not-for-publication``")
$lines.Add("- can publish publicly: ``false``")
$lines.Add("")
$lines.Add("| Decision | Example state | Rationale |")
$lines.Add("| --- | --- | --- |")
foreach ($decision in $decisions) {
  $lines.Add("| ``$($decision.id)`` | ``$($decision.decisionState)`` | $($decision.rationale.Replace("|", "\|")) |")
}
$lines.Add("")
$lines.Add("Run the validator against this example to see that it remains blocked:")
$lines.Add("")
$lines.Add('```powershell')
$lines.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseOwnerApprovalInput.ps1 -InputPath artifacts\final-release\release-owner-approval-input-record.example.json")
$lines.Add('```')

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release owner approval input example written to $jsonPath"
Write-Host "Release owner approval input example written to $markdownPath"
