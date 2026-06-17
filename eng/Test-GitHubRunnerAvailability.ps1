[CmdletBinding()]
param(
  [string]$Repository,
  [Parameter(Mandatory = $true)]
  [string[]]$RequiredLabelSet,
  [switch]$WarnOnly,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  $Repository = $env:GITHUB_REPOSITORY
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  throw "Repository is required. Pass -Repository owner/name or set GITHUB_REPOSITORY."
}

$repositoryParts = $Repository.Split("/", 2)
if ($repositoryParts.Count -ne 2 -or [string]::IsNullOrWhiteSpace($repositoryParts[0]) -or [string]::IsNullOrWhiteSpace($repositoryParts[1])) {
  throw "Repository must use the 'owner/name' format. Value: $Repository"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$ErrorActionPreference = "Stop"

function Expand-LabelSet {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  @(
    foreach ($part in ($Value -split "[,;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $trimmed.ToLowerInvariant()
      }
    }
  ) | Select-Object -Unique
}

function Get-RunnerRows {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryName
  )

  $endpoint = "/repos/$RepositoryName/actions/runners?per_page=100"
  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $lines = @(& gh api $endpoint --paginate --jq '.runners[] | @json' 2>$stderrPath)
    $exitCode = $LASTEXITCODE
    $stderr = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    if ($exitCode -ne 0) {
      throw "Failed to query GitHub Actions runners for '$RepositoryName'. $stderr"
    }

    @(
      foreach ($line in $lines) {
        if ([string]::IsNullOrWhiteSpace($line)) {
          continue
        }

        $runner = $line | ConvertFrom-Json
        $labels = @($runner.labels | ForEach-Object { [string]$_.name })
        [pscustomobject]@{
          id = [string]$runner.id
          name = [string]$runner.name
          os = [string]$runner.os
          status = [string]$runner.status
          busy = [bool]$runner.busy
          labels = $labels
          normalizedLabels = @($labels | ForEach-Object { $_.ToLowerInvariant() })
        }
      }
    )
  }
  finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
  }
}

$runnerRows = @(Get-RunnerRows -RepositoryName $Repository)
$results = New-Object System.Collections.Generic.List[object]
$missing = New-Object System.Collections.Generic.List[object]

foreach ($labelSetText in @($RequiredLabelSet)) {
  if ([string]::IsNullOrWhiteSpace($labelSetText)) {
    continue
  }

  $requiredLabels = @(Expand-LabelSet -Value $labelSetText)
  if ($requiredLabels.Count -eq 0) {
    continue
  }

  $matchingRunners = @(
    $runnerRows | Where-Object {
      $runner = $_
      $labelMatches = @($requiredLabels | Where-Object { $runner.normalizedLabels -contains $_ })
      $labelMatches.Count -eq $requiredLabels.Count
    }
  )
  $onlineMatchingRunners = @($matchingRunners | Where-Object { [string]$_.status -eq "online" })
  $passed = $onlineMatchingRunners.Count -gt 0

  $result = [pscustomobject]@{
    requiredLabels = @($requiredLabels)
    passed = $passed
    matchingRunnerCount = $matchingRunners.Count
    onlineMatchingRunnerCount = $onlineMatchingRunners.Count
    matchingRunners = @($matchingRunners | ForEach-Object {
        [pscustomobject]@{
          name = $_.name
          os = $_.os
          status = $_.status
          busy = $_.busy
          labels = @($_.labels)
        }
      })
  }
  $results.Add($result) | Out-Null

  if (-not $passed) {
    $missing.Add($result) | Out-Null
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\runner-availability"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "github-runner-availability.json"
$markdownPath = Join-Path $outputRoot "github-runner-availability.md"

[pscustomobject]@{
  repository = $Repository
  runnerCount = $runnerRows.Count
  failedCount = $missing.Count
  requestedLabelSets = @($results.ToArray())
  runners = @($runnerRows | ForEach-Object {
      [pscustomobject]@{
        name = $_.name
        os = $_.os
        status = $_.status
        busy = $_.busy
        labels = @($_.labels)
      }
    })
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# GitHub Runner Availability")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("| Required labels | Online match | Matching runners |")
$lines.Add("| --- | --- | --- |")
foreach ($result in $results) {
  $labels = ($result.requiredLabels | ForEach-Object { $codeQuote + $_ + $codeQuote }) -join ", "
  $matchingNames = if ($result.matchingRunners.Count -eq 0) {
    "none"
  }
  else {
    ($result.matchingRunners | ForEach-Object { "$($_.name) ($($_.status))" }) -join ", "
  }
  $lines.Add("| $labels | $($result.passed) | $matchingNames |")
}
$lines.Add("")
$lines.Add("## Runners")
$lines.Add("")
if ($runnerRows.Count -eq 0) {
  $lines.Add("- none")
}
else {
  foreach ($runner in $runnerRows) {
    $lines.Add("- $($runner.name): os=$($runner.os), status=$($runner.status), busy=$($runner.busy), labels=$($runner.labels -join ',')")
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "GitHub runner availability written to $jsonPath"
Write-Host "GitHub runner availability written to $markdownPath"

if ($missing.Count -gt 0) {
  $message = "GitHub runner availability check failed for $($missing.Count) required label set(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
