[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22",
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

$sourceRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
$sourceJsonPath = Join-Path $sourceRoot "linux-runner-evidence-record-template.json"
$sourceMarkdownPath = Join-Path $sourceRoot "linux-runner-evidence-record-template.md"

if (-not (Test-Path -LiteralPath $sourceJsonPath -PathType Leaf)) {
  throw "Linux runner dry-run evidence template was not found: $sourceJsonPath. Run Export-LinuxHandoffIndex.ps1 or the Linux handoff exporter first."
}

$record = Get-Content -LiteralPath $sourceJsonPath -Raw -Encoding utf8 | ConvertFrom-Json

$templateContract = [ordered]@{
  recordKind = "linux-runner-evidence-record-template"
  templateOnly = $true
  runnerOwner = ""
  evidenceRationale = ""
  isRealLinuxRunnerProof = $false
  canPromoteLinuxPackage = $false
}

$record | Add-Member -NotePropertyName "recordKind" -NotePropertyValue $templateContract.recordKind -Force
$record | Add-Member -NotePropertyName "templateOnly" -NotePropertyValue $templateContract.templateOnly -Force
$record | Add-Member -NotePropertyName "runnerOwner" -NotePropertyValue $templateContract.runnerOwner -Force
$record | Add-Member -NotePropertyName "evidenceRationale" -NotePropertyValue $templateContract.evidenceRationale -Force
$record | Add-Member -NotePropertyName "isRealLinuxRunnerProof" -NotePropertyValue $templateContract.isRealLinuxRunnerProof -Force
$record | Add-Member -NotePropertyName "canPromoteLinuxPackage" -NotePropertyValue $templateContract.canPromoteLinuxPackage -Force
$record | Add-Member -NotePropertyName "finalReleaseTemplateState" -NotePropertyValue "template-only" -Force
$record | Add-Member -NotePropertyName "proofClass" -NotePropertyValue "linux-runner-proof" -Force
$record | Add-Member -NotePropertyName "proofClassification" -NotePropertyValue "template-only" -Force
$record | Add-Member -NotePropertyName "expectedValidatedRecord" -NotePropertyValue "artifacts/final-release/linux-runner-evidence-record.json" -Force
$record | Add-Member -NotePropertyName "validatorCommand" -NotePropertyValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-LinuxRunnerEvidenceRecord.ps1 -RuntimePackageKey $RuntimePackageKey -EvidenceRecordPath artifacts/final-release/linux-runner-evidence-record.json -FailOnNotProof" -Force
$record | Add-Member -NotePropertyName "requiredLogFields" -NotePropertyValue @(
  "runner.commands[].logPath",
  "runnerOwner",
  "evidenceRationale",
  "runner.osDescription",
  "runner.kernelVersion",
  "packageOutput.packageConsumerStatus",
  "packageOutput.optionalSmokeStatus"
) -Force
$record | Add-Member -NotePropertyName "requiredSha256Fields" -NotePropertyValue @(
  "packageOutput.runtimeNupkgSha256",
  "commands[].logSha256"
) -Force
$record | Add-Member -NotePropertyName "promotionBoundary" -NotePropertyValue "This final-release template is not Linux runner proof. Windows handoff, template-only, dry-run-only, missing command logs, missing runtime nupkg SHA256, and package-consumer-only evidence cannot promote linux-runner-proof." -Force

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$jsonPath = Join-Path $outputRoot "linux-runner-evidence-record.template.json"
$markdownPath = Join-Path $outputRoot "linux-runner-evidence-record.template.md"

$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Runner Evidence Record Template")
$lines.Add("")
$lines.Add("- runtime package key: ``$RuntimePackageKey``")
$lines.Add("- record kind: ``linux-runner-evidence-record-template``")
$lines.Add("- proof class: ``linux-runner-proof``")
$lines.Add("- proof classification: ``template-only``")
$lines.Add("- final release template state: ``template-only``")
$lines.Add("- is real Linux runner proof: ``False``")
$lines.Add("- can promote Linux package: ``False``")
$lines.Add("- expected validated record: ``artifacts/final-release/linux-runner-evidence-record.json``")
$lines.Add("- validator command: ``$($record.validatorCommand)``")
$lines.Add("")
$lines.Add("## Required Log Fields")
$lines.Add("")
foreach ($field in $record.requiredLogFields) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Required SHA256 Fields")
$lines.Add("")
foreach ($field in $record.requiredSha256Fields) {
  $lines.Add("- ``$field``")
}
$lines.Add("")
$lines.Add("## Promotion Boundary")
$lines.Add("")
$lines.Add($record.promotionBoundary)
$lines.Add("")
$lines.Add("## Source")
$lines.Add("")
$lines.Add("- source JSON: ``artifacts/linux-dry-run/$RuntimePackageKey/linux-runner-evidence-record-template.json``")
if (Test-Path -LiteralPath $sourceMarkdownPath -PathType Leaf) {
  $lines.Add("- source Markdown: ``artifacts/linux-dry-run/$RuntimePackageKey/linux-runner-evidence-record-template.md``")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux runner evidence record template written:"
Write-Host "Linux runner evidence record final-release template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "TemplateState=template-only"
Write-Host "CanPromoteLinuxPackage=False"
