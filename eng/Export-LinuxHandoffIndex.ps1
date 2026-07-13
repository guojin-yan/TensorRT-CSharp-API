[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$root = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
if (-not (Test-Path -LiteralPath $root)) {
  throw "Linux dry-run folder was not found: $root"
}

$items = @(
  Get-ChildItem -LiteralPath $root -File | Sort-Object Name
)

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Handoff Index")
$lines.Add("")
$lines.Add("Runtime key: $RuntimePackageKey")
$lines.Add("")
$lines.Add("## Files")
$lines.Add("")
foreach ($item in $items) {
  $lines.Add("- $($item.Name)")
}
$lines.Add("")
$lines.Add("## Suggested reading order")
$lines.Add("")
$lines.Add("1. README.md")
$lines.Add("2. linux-runtime-dry-run.json")
$lines.Add("3. linux-runner-checklist.md")
$lines.Add("4. linux-preflight-summary.md")
$lines.Add("5. linux-workflow-contract.md")
$lines.Add("6. linux-workflow-contract.json")
$lines.Add("7. linux-package-consumer-plan.md")
$lines.Add("8. linux-package-consumer-plan.json")
$lines.Add("9. linux-runner-execution-status.md")
$lines.Add("10. linux-runner-execution-status.json")
$lines.Add("11. linux-runner-evidence-template.md")
$lines.Add("12. linux-runner-evidence-template.json")
$lines.Add("13. linux-runner-issue-template.md")
$lines.Add("14. linux-runner-evidence-record-template.md")
$lines.Add("15. linux-runner-evidence-record-template.json")
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add("This index is a handoff map. It does not prove that the Linux runtime package has been built or consumed on a Linux x64 runner.")

$markdownPath = Join-Path $root "linux-handoff-index.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux handoff index written to $markdownPath"
