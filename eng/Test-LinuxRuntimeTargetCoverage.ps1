[CmdletBinding()]
param(
  [switch]$RequireFutureTargetPlaceholders,
  [switch]$WarnOnly,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

$expectedDependencyCombos = @(
  "trt8.6-cuda11.8-cudnn8.9",
  "trt8.6-cuda12.1-cudnn8.9",
  "trt10.11-cuda11.8-cudnn8.9",
  "trt10.11-cuda12.9-cudnn9.22",
  "trt11.0-cuda12.9-cudnn9.22",
  "trt11.0-cuda13.2-cudnn9.22"
)

$expectedTargets = @(
  [pscustomobject]@{
    target = "ubuntu22.04-x64-hosted"
    linuxDistro = "ubuntu"
    linuxDistroVersion = "22.04"
    architecture = "x64"
    runnerMode = "hosted"
    expectedCombos = $expectedDependencyCombos
    status = "modeled"
    notes = "Hosted Ubuntu 22.04 x64 is the full six-combination Linux publication line."
  }
  [pscustomobject]@{
    target = "ubuntu24.04-x64-hosted"
    linuxDistro = "ubuntu"
    linuxDistroVersion = "24.04"
    architecture = "x64"
    runnerMode = "hosted"
    expectedCombos = @(
      "trt10.11-cuda12.9-cudnn9.22",
      "trt11.0-cuda12.9-cudnn9.22",
      "trt11.0-cuda13.2-cudnn9.22"
    )
    status = "modeled"
    notes = "Ubuntu 24.04 is limited to modern TensorRT/CUDA combinations available from NVIDIA's Ubuntu 24.04 repositories."
  }
  [pscustomobject]@{
    target = "ubuntu20.04-x64-self-hosted"
    linuxDistro = "ubuntu"
    linuxDistroVersion = "20.04"
    architecture = "x64"
    runnerMode = "self-hosted"
    expectedCombos = @(
      "trt8.6-cuda11.8-cudnn8.9",
      "trt8.6-cuda12.1-cudnn8.9",
      "trt10.11-cuda11.8-cudnn8.9"
    )
    status = "modeled"
    notes = "Ubuntu 20.04 requires a self-hosted Linux x64 runner and remains separate from hosted Ubuntu 22.04/24.04."
  }
)

$futureTargets = @(
  [pscustomobject]@{
    target = "linux-arm64-sbsa"
    status = "future-separate-package-line"
    requiredEvidence = "Dedicated arm64/SBSA runner labels, RID/package IDs, NVIDIA repo architecture, and dependency plan."
  }
  [pscustomobject]@{
    target = "linux-jetson-l4t"
    status = "future-separate-package-line"
    requiredEvidence = "Dedicated Jetson/L4T package IDs, runner/board strategy, L4T-specific NVIDIA dependency plan, and validation evidence."
  }
  [pscustomobject]@{
    target = "non-ubuntu-linux"
    status = "future-separate-package-line"
    requiredEvidence = "Separate package IDs by distro/version plus official NVIDIA dependency source and runner validation."
  }
)

function Get-ComboKey {
  param(
    [Parameter(Mandatory = $true)]
    [object]$Package
  )

  "trt$($Package.tensorRtVersion -replace '^([0-9]+\.[0-9]+).*','$1')-cuda$($Package.cudaVersion)-cudnn$($Package.cudnnVersion -replace '^([0-9]+\.[0-9]+).*','$1')"
}

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$linuxPackages = @($manifest.packages | Where-Object { $_.platform -eq "linux" })

$rows = New-Object System.Collections.Generic.List[object]
$failures = New-Object System.Collections.Generic.List[string]

foreach ($target in $expectedTargets) {
  $packages = @(
    $linuxPackages | Where-Object {
      $_.linuxDistro -eq $target.linuxDistro -and
      $_.linuxDistroVersion -eq $target.linuxDistroVersion -and
      $_.architecture -eq $target.architecture -and
      $_.runnerMode -eq $target.runnerMode
    }
  )

  $actualCombos = @($packages | ForEach-Object { Get-ComboKey -Package $_ } | Sort-Object -Unique)
  $missingCombos = @($target.expectedCombos | Where-Object { $actualCombos -notcontains $_ })
  $unexpectedCombos = @($actualCombos | Where-Object { $target.expectedCombos -notcontains $_ })
  $ambiguousKeys = @($packages | Where-Object { $_.key -notmatch "^linux-$($target.architecture)-$($target.linuxDistro)$([regex]::Escape($target.linuxDistroVersion))-trt" })
  $passed = $missingCombos.Count -eq 0 -and $unexpectedCombos.Count -eq 0 -and $ambiguousKeys.Count -eq 0

  $rows.Add([pscustomobject]@{
      target = $target.target
      status = $target.status
      expectedCount = $target.expectedCombos.Count
      actualCount = $actualCombos.Count
      expectedCombos = @($target.expectedCombos)
      actualCombos = @($actualCombos)
      missingCombos = @($missingCombos)
      unexpectedCombos = @($unexpectedCombos)
      packageKeys = @($packages | Select-Object -ExpandProperty key)
      passed = $passed
      notes = $target.notes
    }) | Out-Null

  if (-not $passed) {
    $failures.Add("$($target.target): missing=[$($missingCombos -join ',')] unexpected=[$($unexpectedCombos -join ',')] ambiguousKeys=$($ambiguousKeys.Count)") | Out-Null
  }
}

$publishedFuturePackages = @(
  $linuxPackages | Where-Object {
    $_.architecture -ne "x64" -or
    $_.linuxDistro -ne "ubuntu"
  }
)

if ($RequireFutureTargetPlaceholders.IsPresent -and $publishedFuturePackages.Count -gt 0) {
  $failures.Add("Future Linux targets must remain separate until explicitly modeled; found package keys: $($publishedFuturePackages.key -join ', ')") | Out-Null
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-target-coverage"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "linux-runtime-target-coverage.json"
$markdownPath = Join-Path $outputRoot "linux-runtime-target-coverage.md"

[pscustomobject]@{
  expectedDependencyCombinations = @($expectedDependencyCombos)
  failedCount = $failures.Count
  modeledTargets = @($rows.ToArray())
  futureTargets = @($futureTargets)
  publishedFuturePackageKeys = @($publishedFuturePackages | Select-Object -ExpandProperty key)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Linux Runtime Target Coverage")
$lines.Add("")
$lines.Add("| Target | Status | Expected | Actual | Passed |")
$lines.Add("| --- | --- | ---: | ---: | --- |")
foreach ($row in $rows) {
  $lines.Add("| " + $codeQuote + $row.target + $codeQuote + " | " + $codeQuote + $row.status + $codeQuote + " | $($row.expectedCount) | $($row.actualCount) | $($row.passed) |")
}

$lines.Add("")
$lines.Add("## Modeled Targets")
foreach ($row in $rows) {
  $lines.Add("")
  $lines.Add("### " + $row.target)
  $lines.Add("")
  $lines.Add("- expected combinations: " + (($row.expectedCombos | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  $lines.Add("- actual combinations: " + (($row.actualCombos | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  $lines.Add("- package keys: " + (($row.packageKeys | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  $lines.Add("- notes: $($row.notes)")
}

$lines.Add("")
$lines.Add("## Future Separate Package Lines")
foreach ($futureTarget in $futureTargets) {
  $lines.Add("- " + $codeQuote + $futureTarget.target + $codeQuote + ": " + $futureTarget.requiredEvidence)
}

$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Failed checks: $($failures.Count)")
$lines.Add("- Future target packages currently published in manifest: $($publishedFuturePackages.Count)")

if ($failures.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Failures")
  foreach ($failure in $failures) {
    $lines.Add("- $failure")
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux runtime target coverage written to $jsonPath"
Write-Host "Linux runtime target coverage written to $markdownPath"

if ($failures.Count -gt 0) {
  $message = "Linux runtime target coverage has $($failures.Count) failed check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
