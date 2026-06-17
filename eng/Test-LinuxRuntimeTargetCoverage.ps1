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

$targetCatalogPath = Join-Path $RepositoryRoot "pack\runtime\linux-runtime-targets.manifest.json"
if (-not (Test-Path -LiteralPath $targetCatalogPath -PathType Leaf)) {
  throw "Linux runtime target catalog was not found: $targetCatalogPath"
}

$targetCatalog = Get-Content -LiteralPath $targetCatalogPath -Raw -Encoding utf8 | ConvertFrom-Json
$expectedDependencyCombos = @($targetCatalog.dependencyCombinations | ForEach-Object { [string]$_ })

$expectedTargets = @(
  foreach ($target in @($targetCatalog.targets)) {
    $expectedCombos = @($target.expectedCombinations | ForEach-Object { [string]$_ })
    if ($expectedCombos -contains "all") {
      $expectedCombos = $expectedDependencyCombos
    }

    [pscustomobject]@{
      target = [string]$target.target
      linuxDistro = [string]$target.linuxDistro
      linuxDistroVersion = [string]$target.linuxDistroVersion
      architecture = [string]$target.architecture
      runnerMode = [string]$target.runnerMode
      expectedCombos = @($expectedCombos)
      status = [string]$target.status
      publicationRequirement = [string]$target.publicationRequirement
      keySetAliases = @($target.keySetAliases | ForEach-Object { [string]$_ })
      notes = [string]$target.notes
    }
  }
)

$futureTargets = @($targetCatalog.futureTargets)

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
      publicationRequirement = $target.publicationRequirement
      keySetAliases = @($target.keySetAliases)
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
  targetCatalogPath = $targetCatalogPath
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
$lines.Add("| Target | Status | Requirement | Expected | Actual | Passed |")
$lines.Add("| --- | --- | --- | ---: | ---: | --- |")
foreach ($row in $rows) {
  $lines.Add("| " + $codeQuote + $row.target + $codeQuote + " | " + $codeQuote + $row.status + $codeQuote + " | " + $codeQuote + $row.publicationRequirement + $codeQuote + " | $($row.expectedCount) | $($row.actualCount) | $($row.passed) |")
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
  $lines.Add("- key set aliases: " + (($row.keySetAliases | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "))
  $lines.Add("- notes: $($row.notes)")
}

$lines.Add("")
$lines.Add("## Future Separate Package Lines")
foreach ($futureTarget in $futureTargets) {
  $lines.Add("- " + $codeQuote + $futureTarget.target + $codeQuote + ": " + $futureTarget.requiredEvidence)
  foreach ($evidenceItem in @($futureTarget.requiredEvidenceItems)) {
    $lines.Add("  - required evidence: $evidenceItem")
  }
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
