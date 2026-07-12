[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
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

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ($package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$runnerOwnedEvidence = @(
  "Linux x64 host identity and runner labels.",
  "TensorRT root, CUDA root, and cuDNN root used by the run.",
  "Validate-LinuxRuntimeInputs output.",
  "CMake configure log for $($package.buildPreset).",
  "CMake build log for $($package.buildPreset).",
  "Collect-RuntimeAssets output and copied .so inventory.",
  "Runtime package nupkg path and SHA256.",
  "Package consumer restore/build/native-copy summary.",
  "Optional GPU smoke summary if driver, CUDA runtime, and GPU access are available."
)

$commands = @(
  "pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey $RuntimePackageKey",
  "cmake --preset $($package.buildPreset)",
  "cmake --build --preset $($package.buildPreset) --parallel",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey $RuntimePackageKey",
  "dotnet pack ./pack/runtime/$RuntimePackageKey/$($package.packageId).csproj -c Release -o ./artifacts/runtime-nupkg",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey",
  "pwsh -NoProfile -ExecutionPolicy Bypass -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey $RuntimePackageKey -RunSmoke"
)

$template = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runtimeKey = [string]$package.key
  packageId = [string]$package.packageId
  rid = [string]$package.rid
  validationState = [string]$package.validationState
  distro = "$($package.linuxDistro)$($package.linuxDistroVersion)"
  buildPreset = [string]$package.buildPreset
  evidenceState = "template-only"
  isRealLinuxRunnerProof = $false
  handoffEvidenceMayBeGeneratedOnWindows = $true
  runnerOwnedEvidence = $runnerOwnedEvidence
  commands = $commands
  promotionRules = @(
    "Do not promote dry-run-only to local-validated from this template alone.",
    "Promotion requires a Linux x64 runner to execute the build, package, and consumer commands.",
    "GPU smoke is optional for packaging proof but required before claiming runtime smoke passed.",
    "Real callback runtime proof remains separate and requires InvocationCount>0 plus IsRealCallbackRuntimeProof=True."
  )
}

$jsonPath = Join-Path $outputRoot "linux-runner-evidence-template.json"
$markdownPath = Join-Path $outputRoot "linux-runner-evidence-template.md"
$issuePath = Join-Path $outputRoot "linux-runner-issue-template.md"
$template | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Runner Evidence Template")
$lines.Add("")
$lines.Add("Runtime key: ``$($package.key)``")
$lines.Add("")
$lines.Add("Package ID: ``$($package.packageId)``")
$lines.Add("")
$lines.Add("Evidence state: ``template-only``")
$lines.Add("")
$lines.Add("This file is a handoff template. It is not Linux runner proof until a Linux x64 runner fills the evidence fields with real command outputs.")
$lines.Add("")
$lines.Add("## Runner-Owned Evidence")
$lines.Add("")
foreach ($item in $runnerOwnedEvidence) {
  $lines.Add("- [ ] $item")
}
$lines.Add("")
$lines.Add("## Command Order")
$lines.Add("")
foreach ($command in $commands) {
  $lines.Add("1. ``$command``")
}
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $template.promotionRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Evidence Fill-In")
$lines.Add("")
$lines.Add("- runner OS:")
$lines.Add("- runner name:")
$lines.Add("- TensorRT root:")
$lines.Add("- CUDA root:")
$lines.Add("- cuDNN root:")
$lines.Add("- runtime nupkg:")
$lines.Add("- runtime nupkg SHA256:")
$lines.Add("- package consumer status:")
$lines.Add("- optional smoke status:")
$lines.Add("- release owner notes:")

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

$issueLines = New-Object System.Collections.Generic.List[string]
$issueLines.Add("# Linux Runner Validation Issue Template")
$issueLines.Add("")
$issueLines.Add("Runtime key: ``$($package.key)``")
$issueLines.Add("")
$issueLines.Add("Package ID: ``$($package.packageId)``")
$issueLines.Add("")
$issueLines.Add("Evidence state before runner execution: ``template-only``")
$issueLines.Add("")
$issueLines.Add("This issue template must be filled by the Linux x64 runner owner. It is not proof until real command output, package paths, and runner metadata are attached.")
$issueLines.Add("")
$issueLines.Add("## Owner Checklist")
$issueLines.Add("")
$issueLines.Add("- [ ] Runner is Linux x64 and records OS image, kernel, CPU architecture, and runner labels.")
$issueLines.Add("- [ ] TensorRT root, CUDA root, and cuDNN root are recorded.")
$issueLines.Add("- [ ] ``Validate-LinuxRuntimeInputs.ps1`` output is attached.")
$issueLines.Add("- [ ] CMake configure output for ``$($package.buildPreset)`` is attached.")
$issueLines.Add("- [ ] CMake build output for ``$($package.buildPreset)`` is attached.")
$issueLines.Add("- [ ] Runtime asset collection output and copied ``.so`` inventory are attached.")
$issueLines.Add("- [ ] Runtime ``.nupkg`` path and SHA256 are attached.")
$issueLines.Add("- [ ] Package consumer restore/build/native-copy summary is attached.")
$issueLines.Add("- [ ] Optional GPU smoke output is attached only if driver, CUDA runtime, and GPU access are compatible.")
$issueLines.Add("")
$issueLines.Add("## Commands")
$issueLines.Add("")
foreach ($command in $commands) {
  $issueLines.Add('```bash')
  $issueLines.Add($command)
  $issueLines.Add('```')
}
$issueLines.Add("")
$issueLines.Add("## Required Attachments")
$issueLines.Add("")
$issueLines.Add("- ``linux-runner-evidence-template.md`` filled with runner-owned evidence.")
$issueLines.Add("- ``linux-runner-evidence-template.json`` updated or accompanied by equivalent structured evidence.")
$issueLines.Add("- Configure/build logs.")
$issueLines.Add("- Runtime nupkg SHA256.")
$issueLines.Add("- Package consumer validation summary.")
$issueLines.Add("")
$issueLines.Add("## Boundary Notes")
$issueLines.Add("")
$issueLines.Add("- Keep ``isRealLinuxRunnerProof=false`` until a Linux x64 runner has attached real evidence.")
$issueLines.Add("- Do not treat ``dry-run-only`` or Windows-generated handoff files as Linux runner proof.")
$issueLines.Add("- Do not treat ``blocked-by-cuda-driver`` as smoke passed.")
$issueLines.Add("- Real callback runtime proof remains separate and still requires ``InvocationCount>0`` plus ``IsRealCallbackRuntimeProof=True``.")

$issueLines | Set-Content -LiteralPath $issuePath -Encoding utf8

Write-Host "Linux runner evidence template written to $jsonPath"
Write-Host "Linux runner evidence template written to $markdownPath"
Write-Host "Linux runner issue template written to $issuePath"
