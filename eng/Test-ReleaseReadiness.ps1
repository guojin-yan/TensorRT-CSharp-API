[CmdletBinding()]
param(
  [string]$Repository = "guojin-yan/TensorRT-CSharp-API",
  [AllowEmptyString()]
  [string]$NuGetApiKeyAvailable,
  [AllowEmptyString()]
  [string]$RunnerAuditTokenAvailable,
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

function Test-GhAuthAvailable {
  $stderrPath = [IO.Path]::GetTempFileName()
  try {
    $null = & gh auth status -h github.com 2>$stderrPath
    return $LASTEXITCODE -eq 0
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

function Get-RunnerRows {
  param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryName,
    [AllowEmptyString()]
    [string]$GitHubToken
  )

  $result = Invoke-GhJsonLines -Arguments @("api", "/repos/$RepositoryName/actions/runners?per_page=100", "--paginate", "--jq", ".runners[] | @json") -GitHubToken $GitHubToken -AllowFailure
  if (-not $result.success) {
    return [pscustomobject]@{
      success = $false
      rows = @()
      stderr = $result.stderr
    }
  }

  $rows = @(
    foreach ($runner in @($result.items)) {
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

  [pscustomobject]@{
    success = $true
    rows = @($rows)
    stderr = ""
  }
}

function Test-RunnerLabelSet {
  param(
    [object[]]$RunnerRows,
    [Parameter(Mandatory = $true)]
    [string]$RequiredLabelSet
  )

  $requiredLabels = @(Expand-LabelSet -Value $RequiredLabelSet)
  $matchingRunners = @(
    $RunnerRows | Where-Object {
      $runner = $_
      $matches = @($requiredLabels | Where-Object { $runner.normalizedLabels -contains $_ })
      $matches.Count -eq $requiredLabels.Count
    }
  )
  $onlineMatchingRunners = @($matchingRunners | Where-Object { [string]$_.status -eq "online" })

  [pscustomobject]@{
    requiredLabelSet = $RequiredLabelSet
    requiredLabels = @($requiredLabels)
    passed = $onlineMatchingRunners.Count -gt 0
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
}

$hasNuGetApiKey = ConvertTo-BoolOrNull -Value $NuGetApiKeyAvailable
$runnerAuditTokenIsAvailable = ConvertTo-BoolOrNull -Value $RunnerAuditTokenAvailable

if ($null -eq $hasNuGetApiKey -or $null -eq $runnerAuditTokenIsAvailable) {
  $secretResult = Invoke-GhJsonLines -Arguments @("secret", "list", "--repo", $Repository, "--json", "name", "--jq", ".[] | @json") -AllowFailure
  if ($secretResult.success) {
    $secretNames = @($secretResult.items | ForEach-Object { [string]$_.name })
    if ($null -eq $hasNuGetApiKey) {
      $hasNuGetApiKey = $secretNames -contains "NUGET_API_KEY"
    }

    if ($null -eq $runnerAuditTokenIsAvailable) {
      $runnerAuditTokenIsAvailable = $secretNames -contains "RUNNER_AUDIT_TOKEN"
    }
  }
  else {
    if ($null -eq $hasNuGetApiKey) {
      $hasNuGetApiKey = $false
    }

    if ($null -eq $runnerAuditTokenIsAvailable) {
      $runnerAuditTokenIsAvailable = $false
    }
  }
}

$runnerRows = @()
$runnerQueryAttempted = $false
$runnerQuerySucceeded = $false
$runnerQuerySource = ""
$runnerQueryDetail = ""

$runnerQueryToken = ""
if (-not [string]::IsNullOrWhiteSpace($env:RUNNER_AUDIT_TOKEN)) {
  $runnerQuerySource = "RUNNER_AUDIT_TOKEN"
  $runnerQueryToken = $env:RUNNER_AUDIT_TOKEN
}
elseif (-not [string]::IsNullOrWhiteSpace($env:GH_TOKEN) -and -not ([string]::Equals($env:GITHUB_ACTIONS, "true", [System.StringComparison]::OrdinalIgnoreCase) -and -not $runnerAuditTokenIsAvailable)) {
  $runnerQuerySource = "GH_TOKEN"
}
elseif (-not [string]::Equals($env:GITHUB_ACTIONS, "true", [System.StringComparison]::OrdinalIgnoreCase) -and (Test-GhAuthAvailable)) {
  $runnerQuerySource = "gh-auth"
}

if (-not [string]::IsNullOrWhiteSpace($runnerQuerySource)) {
  $runnerQueryAttempted = $true
  $runnerResult = Get-RunnerRows -RepositoryName $Repository -GitHubToken $runnerQueryToken
  $runnerQuerySucceeded = $runnerResult.success
  $runnerRows = @($runnerResult.rows)
  $runnerQueryDetail = if ($runnerResult.success) { "source=$runnerQuerySource; queried $($runnerRows.Count) runner(s)." } else { "source=$runnerQuerySource; $($runnerResult.stderr)" }
}
else {
  $runnerQueryDetail = "not attempted; RUNNER_AUDIT_TOKEN is not available, no usable local gh auth was found, and GitHub Actions' default token cannot query runner availability reliably."
}

$runnerChecks = @()
if ($runnerQuerySucceeded) {
  $runnerChecks = @(
    Test-RunnerLabelSet -RunnerRows $runnerRows -RequiredLabelSet "self-hosted,windows,x64"
  )
}

$futureTargetReadiness = @(
  [pscustomobject]@{
    area = "future-target:linux-arm64-sbsa"
    gate = "advisory"
    target = "linux-arm64-sbsa"
    ready = $false
    reason = "Future package line only: needs dedicated arm64/SBSA package IDs, runner labels, NVIDIA repository architecture, dependency plan, and consumer validation."
  }
  [pscustomobject]@{
    area = "future-target:linux-jetson-l4t"
    gate = "advisory"
    target = "linux-jetson-l4t"
    ready = $false
    reason = "Future package line only: needs Jetson/L4T package IDs, board or runner strategy, L4T dependency source, bridge/runtime validation, and consumer validation."
  }
  [pscustomobject]@{
    area = "future-target:non-ubuntu-linux"
    gate = "advisory"
    target = "non-ubuntu-linux"
    ready = $false
    reason = "Future package line only: needs distro/version package IDs, runner image, official NVIDIA dependency source, pinned dependency plan, and consumer validation."
  }
)

$readiness = New-Object System.Collections.Generic.List[object]
$readiness.Add([pscustomobject]@{
    area = "managed-nuget-org"
    gate = "required"
    ready = [bool]$hasNuGetApiKey
    detail = if ($hasNuGetApiKey) { "NUGET_API_KEY is available." } else { "NUGET_API_KEY is missing; managed package cannot be published to nuget.org from GitHub Actions." }
  }) | Out-Null
$readiness.Add([pscustomobject]@{
    area = "runner-audit"
    gate = "advisory"
    ready = [bool]$runnerAuditTokenIsAvailable
    detail = if ($runnerAuditTokenIsAvailable) { "RUNNER_AUDIT_TOKEN is available." } else { "RUNNER_AUDIT_TOKEN is missing; GitHub Actions runner availability audit stays warning-only unless local gh-auth can query it." }
  }) | Out-Null
$readiness.Add([pscustomobject]@{
    area = "runner-query"
    gate = "advisory"
    ready = [bool]$runnerQuerySucceeded
    detail = $runnerQueryDetail
  }) | Out-Null

foreach ($runnerCheck in $runnerChecks) {
  $readiness.Add([pscustomobject]@{
      area = "runner:$($runnerCheck.requiredLabelSet)"
      gate = "advisory"
      ready = [bool]$runnerCheck.passed
      detail = "onlineMatchingRunnerCount=$($runnerCheck.onlineMatchingRunnerCount); matchingRunnerCount=$($runnerCheck.matchingRunnerCount)"
    }) | Out-Null
}

foreach ($futureTarget in $futureTargetReadiness) {
  $readiness.Add([pscustomobject]@{
      area = $futureTarget.area
      gate = $futureTarget.gate
      ready = [bool]$futureTarget.ready
      detail = $futureTarget.reason
    }) | Out-Null
}

$notReady = @($readiness | Where-Object { -not $_.ready })
$blockingNotReady = @($notReady | Where-Object { [string]$_.gate -eq "required" })
$advisoryNotReady = @($notReady | Where-Object { [string]$_.gate -ne "required" })
$outputRoot = Join-Path $RepositoryRoot "artifacts\release-readiness"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "release-readiness.json"
$markdownPath = Join-Path $outputRoot "release-readiness.md"

[pscustomobject]@{
  repository = $Repository
  readyCount = @($readiness | Where-Object { $_.ready }).Count
  notReadyCount = $notReady.Count
  blockingNotReadyCount = $blockingNotReady.Count
  advisoryNotReadyCount = $advisoryNotReady.Count
  readiness = @($readiness.ToArray())
  runnerAuditTokenAvailable = [bool]$runnerAuditTokenIsAvailable
  nugetApiKeyAvailable = [bool]$hasNuGetApiKey
  runnerQueryAttempted = [bool]$runnerQueryAttempted
  runnerQuerySucceeded = [bool]$runnerQuerySucceeded
  runnerQuerySource = $runnerQuerySource
  runnerQueryDetail = $runnerQueryDetail
  runners = @($runnerRows | ForEach-Object {
      [pscustomobject]@{
        name = $_.name
        os = $_.os
        status = $_.status
        busy = $_.busy
        labels = @($_.labels)
      }
    })
  futureTargets = @($futureTargetReadiness)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$codeQuote = [string][char]96
$lines.Add("# Release Readiness")
$lines.Add("")
$lines.Add("Repository: " + $codeQuote + $Repository + $codeQuote)
$lines.Add("")
$lines.Add("| Area | Gate | Ready | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $readiness) {
  $detail = ([string]$item.detail).Replace("|", "\|")
  $lines.Add("| " + $codeQuote + $item.area + $codeQuote + " | $($item.gate) | $($item.ready) | $detail |")
}
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("- Ready checks: $(@($readiness | Where-Object { $_.ready }).Count)")
$lines.Add("- Not-ready checks: $($notReady.Count)")
$lines.Add("- Blocking not-ready checks: $($blockingNotReady.Count)")
$lines.Add("- Advisory not-ready checks: $($advisoryNotReady.Count)")
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Release readiness written to $jsonPath"
Write-Host "Release readiness written to $markdownPath"

if ($blockingNotReady.Count -gt 0) {
  $message = "Release readiness has $($blockingNotReady.Count) blocking not-ready check(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
