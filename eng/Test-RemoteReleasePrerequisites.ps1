[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [AllowEmptyString()]
  [string]$NuGetApiKeyAvailable,
  [AllowEmptyString()]
  [string]$RunnerAuditTokenAvailable,
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

function ConvertTo-BoolOrNull {
  param(
    [AllowEmptyString()]
    [string]$Value
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $null
  }

  $Value -in @("1", "true", "True", "TRUE", "yes", "Yes", "YES")
}

function Expand-TokenList {
  param(
    [string[]]$Values
  )

  $tokens = New-Object System.Collections.Generic.List[string]
  foreach ($value in @($Values)) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($part in ($value -split "[`r`n;]")) {
      $trimmed = $part.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $tokens.Add($trimmed)
      }
    }
  }

  @($tokens | Select-Object -Unique)
}

function Invoke-GhJsonLines {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$Arguments,
    [AllowEmptyString()]
    [string]$GitHubToken,
    [switch]$AllowFailure
  )

  $stderrPath = [IO.Path]::GetTempFileName()
  $previousGhToken = $env:GH_TOKEN
  try {
    if (-not [string]::IsNullOrWhiteSpace($GitHubToken)) {
      $env:GH_TOKEN = $GitHubToken
    }

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
    if ($null -eq $previousGhToken) {
      Remove-Item Env:\GH_TOKEN -ErrorAction SilentlyContinue
    }
    else {
      $env:GH_TOKEN = $previousGhToken
    }

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

$requiredSecrets = @(Expand-TokenList -Values $RequiredSecret)
$requiredRunnerLabelSets = @(Expand-TokenList -Values $RequiredRunnerLabelSet)

$knownSecretAvailability = @{}
$nugetApiKeyIsAvailable = ConvertTo-BoolOrNull -Value $NuGetApiKeyAvailable
$runnerAuditTokenIsAvailable = ConvertTo-BoolOrNull -Value $RunnerAuditTokenAvailable
if ($null -ne $nugetApiKeyIsAvailable) {
  $knownSecretAvailability["NUGET_API_KEY"] = [bool]$nugetApiKeyIsAvailable
}
if ($null -ne $runnerAuditTokenIsAvailable) {
  $knownSecretAvailability["RUNNER_AUDIT_TOKEN"] = [bool]$runnerAuditTokenIsAvailable
}

$unknownRequiredSecrets = @($requiredSecrets | Where-Object { -not $knownSecretAvailability.ContainsKey($_) })
$secretResult = if ($unknownRequiredSecrets.Count -gt 0) {
  Invoke-GhJsonLines -Arguments @("secret", "list", "--repo", $Repository, "--json", "name", "--jq", ".[] | @json") -AllowFailure
}
else {
  [pscustomobject]@{
    success = $true
    items = @()
    stderr = ""
  }
}
$secretNames = if ($secretResult.success) { @($secretResult.items | ForEach-Object { [string]$_.name }) } else { @() }

$runnerQuerySource = ""
$runnerQueryToken = ""
if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_AUDIT_TOKEN)) {
  $runnerQuerySource = "RUNNER_AUDIT_TOKEN"
  $runnerQueryToken = $env:RUNNER_AUDIT_TOKEN
}
elseif (-not [string]::Equals($env:GITHUB_ACTIONS, "true", [System.StringComparison]::OrdinalIgnoreCase)) {
  $runnerQuerySource = "gh-auth"
}

$runnerResult = if ([string]::IsNullOrWhiteSpace($runnerQuerySource)) {
  [pscustomobject]@{
    success = $false
    items = @()
    stderr = "RUNNER_AUDIT_TOKEN is not set; GitHub Actions' default token cannot query repository self-hosted runners."
  }
}
else {
  Invoke-GhJsonLines -Arguments @("api", "/repos/$Repository/actions/runners?per_page=100", "--paginate", "--jq", ".runners[] | @json") -GitHubToken $runnerQueryToken -AllowFailure
}
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
  foreach ($name in $requiredSecrets) {
    $present = if ($knownSecretAvailability.ContainsKey($name)) {
      [bool]$knownSecretAvailability[$name]
    }
    elseif ($secretResult.success) {
      $secretNames -contains $name
    }
    else {
      $false
    }

    [pscustomobject]@{
      name = $name
      present = $present
      source = if ($knownSecretAvailability.ContainsKey($name)) { "input" } elseif ($secretResult.success) { "gh-secret-list" } else { "query-failed" }
    }
  }
)

$runnerChecks = @(
  foreach ($labelSet in $requiredRunnerLabelSets) {
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
  if ($runnerResult.success -and -not $check.ready) {
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
  runnerQuerySource = $runnerQuerySource
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
