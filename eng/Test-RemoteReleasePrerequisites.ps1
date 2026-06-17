[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [string[]]$RequiredSecret = @("NUGET_API_KEY", "RUNNER_AUDIT_TOKEN"),
  [string[]]$RequiredRunnerLabelSet = @(
    "self-hosted,windows,x64",
    "self-hosted,linux,x64,ubuntu-20.04"
  ),
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

function Invoke-GhJsonLines {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments,
    [switch]$AllowFailure
  )

  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $output = @(& gh @Arguments 2>$stderrPath)
    $exitCode = $LASTEXITCODE
    $stderr = if (Test-Path -LiteralPath $stderrPath -PathType Leaf) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    if ($exitCode -ne 0) {
      if ($AllowFailure.IsPresent) {
        return [pscustomobject]@{
          success = $false
          items = @()
          stderr = $stderr
        }
      }

      throw "gh $($Arguments -join ' ') failed. $stderr"
    }

    $items = @(
      foreach ($line in $output) {
        if ([string]::IsNullOrWhiteSpace($line)) {
          continue
        }

        $line | ConvertFrom-Json
      }
    )

    [pscustomobject]@{
      success = $true
      items = $items
      stderr = $stderr
    }
  }
  finally {
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
  }
}

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

$secretResult = Invoke-GhJsonLines -Arguments @("secret", "list", "--repo", $Repository, "--json", "name", "--jq", ".[] | @json") -AllowFailure
$secretNames = if ($secretResult.success) { @($secretResult.items | ForEach-Object { [string]$_.name }) } else { @() }

$runnerResult = Invoke-GhJsonLines -Arguments @("api", "/repos/$Repository/actions/runners?per_page=100", "--paginate", "--jq", ".runners[] | @json") -AllowFailure
$runnerRows = if ($runnerResult.success) {
  @(
    foreach ($runner in $runnerResult.items) {
      $labels = @($runner.labels | ForEach-Object { [string]$_.name })
      [pscustomobject]@{
        name = [string]$runner.name
        os = [string]$runner.os
        status = [string]$runner.status
        busy = [bool]$runner.busy
        labels = @($labels)
        normalizedLabels = @($labels | ForEach-Object { $_.ToLowerInvariant() })
      }
    }
  )
}
else {
  @()
}

$secretChecks = @(
  foreach ($name in $RequiredSecret) {
    [pscustomobject]@{
      name = $name
      present = $secretNames -contains $name
    }
  }
)

$runnerChecks = @(
  foreach ($labelSet in $RequiredRunnerLabelSet) {
    $requiredLabels = @(Expand-LabelSet -Value $labelSet)
    $matching = @(
      $runnerRows | Where-Object {
        $runner = $_
        @($requiredLabels | Where-Object { $runner.normalizedLabels -contains $_ }).Count -eq $requiredLabels.Count
      }
    )
    $online = @($matching | Where-Object { [string]$_.status -eq "online" })

    [pscustomobject]@{
      requiredLabelSet = $labelSet
      matchingRunnerCount = $matching.Count
      onlineMatchingRunnerCount = $online.Count
      ready = $online.Count -gt 0
      matchingRunners = @($matching | ForEach-Object {
          [pscustomobject]@{
            name = $_.name
            os = $_.os
            status = $_.status
            busy = $_.busy
            labels = @($_.labels)
          }
        })
    }
  }
)

$failures = New-Object System.Collections.Generic.List[string]
foreach ($check in $secretChecks) {
  if (-not $check.present) {
    $failures.Add("Missing repository secret: $($check.name)") | Out-Null
  }
}

if (-not $runnerResult.success) {
  $failures.Add("Unable to query repository runners: $($runnerResult.stderr)") | Out-Null
}

foreach ($check in $runnerChecks) {
  if (-not $check.ready) {
    $failures.Add("No online runner for label set: $($check.requiredLabelSet)") | Out-Null
  }
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\remote-release-prerequisites"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "remote-release-prerequisites.json"
$markdownPath = Join-Path $outputRoot "remote-release-prerequisites.md"

[pscustomobject]@{
  repository = $Repository
  failedCount = $failures.Count
  secrets = @($secretChecks)
  runnerQuerySucceeded = [bool]$runnerResult.success
  runnerQueryError = if ($runnerResult.success) { "" } else { $runnerResult.stderr }
  runners = @($runnerRows | ForEach-Object {
      [pscustomobject]@{
        name = $_.name
        os = $_.os
        status = $_.status
        busy = $_.busy
        labels = @($_.labels)
      }
    })
  runnerChecks = @($runnerChecks)
  failures = @($failures.ToArray())
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Remote Release Prerequisites")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("## Secrets")
$lines.Add("")
$lines.Add("| Secret | Present |")
$lines.Add("| --- | --- |")
foreach ($check in $secretChecks) {
  $lines.Add("| " + $codeQuote + $check.name + $codeQuote + " | $($check.present) |")
}
$lines.Add("")
$lines.Add("## Runners")
$lines.Add("")
$lines.Add("| Required labels | Ready | Online matches | Total matches |")
$lines.Add("| --- | --- | ---: | ---: |")
foreach ($check in $runnerChecks) {
  $lines.Add("| " + $codeQuote + $check.requiredLabelSet + $codeQuote + " | $($check.ready) | $($check.onlineMatchingRunnerCount) | $($check.matchingRunnerCount) |")
}

if ($failures.Count -gt 0) {
  $lines.Add("")
  $lines.Add("## Failures")
  foreach ($failure in $failures) {
    $lines.Add("- $failure")
  }
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Remote release prerequisites written to $jsonPath"
Write-Host "Remote release prerequisites written to $markdownPath"

if ($failures.Count -gt 0) {
  $message = "Remote release prerequisites have $($failures.Count) failed check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
