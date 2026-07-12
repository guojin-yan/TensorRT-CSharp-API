[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Invoke-GitLines {
  param([string[]]$Arguments)

  $output = & git @Arguments 2>$null
  if ($LASTEXITCODE -ne 0) {
    return @()
  }

  return @($output)
}

function Test-GitTrackedPath {
  param([string]$RelativePath)

  & git ls-files --error-unmatch $RelativePath *> $null
  return $LASTEXITCODE -eq 0
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

Push-Location $RepositoryRoot
try {
  $trackedTests = @(Invoke-GitLines -Arguments @("ls-files", "tests/JYPPX.ProjectQuality.Tests/*.cs"))
  $scriptRefs = New-Object System.Collections.Generic.List[object]
  foreach ($testFile in $trackedTests) {
    $path = Join-Path $RepositoryRoot $testFile
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      continue
    }

    $text = Get-Content -LiteralPath $path -Raw -Encoding utf8
    foreach ($match in [regex]::Matches($text, '([A-Za-z0-9_.-]+\.ps1)')) {
      $scriptName = $match.Groups[1].Value
      $scriptPath = Join-Path "eng" $scriptName
      $scriptRefs.Add([pscustomobject]@{
          testFile = $testFile
          scriptName = $scriptName
          scriptPath = $scriptPath
        })
    }
  }

  $uniqueScriptPaths = @($scriptRefs | Sort-Object scriptPath -Unique)
  $trackedScriptReferences = New-Object System.Collections.Generic.List[object]
  $untrackedScriptReferences = New-Object System.Collections.Generic.List[object]
  $missingScriptFiles = New-Object System.Collections.Generic.List[object]

  foreach ($reference in $uniqueScriptPaths) {
    $scriptPath = [string]$reference.scriptPath
    $exists = Test-Path -LiteralPath (Join-Path $RepositoryRoot $scriptPath) -PathType Leaf
    $tracked = Test-GitTrackedPath -RelativePath $scriptPath
    $referencingTests = @($scriptRefs | Where-Object { $_.scriptPath -eq $scriptPath } | Select-Object -ExpandProperty testFile -Unique)
    $entry = [pscustomobject]@{
      scriptPath = $scriptPath
      exists = $exists
      tracked = $tracked
      referencingTestCount = [int]$referencingTests.Count
      referencingTests = @($referencingTests)
    }

    if ($tracked) {
      $trackedScriptReferences.Add($entry)
    }
    else {
      $untrackedScriptReferences.Add($entry)
      if (-not $exists) {
        $missingScriptFiles.Add($entry)
      }
    }
  }

  $topUntrackedReferences = @($untrackedScriptReferences |
    Sort-Object @{Expression = "referencingTestCount"; Descending = $true}, @{Expression = "scriptPath"; Descending = $false} |
    Select-Object -First 50)

  $record = [pscustomobject]@{
    recordKind = "release-script-reference-index"
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    indexState = "release-script-reference-index-ready-non-proof"
    trackedTestFileCount = [int]$trackedTests.Count
    totalScriptReferenceCount = [int]$scriptRefs.Count
    uniqueScriptReferenceCount = [int]$uniqueScriptPaths.Count
    trackedScriptReferenceCount = [int]$trackedScriptReferences.Count
    untrackedScriptReferenceCount = [int]$untrackedScriptReferences.Count
    missingScriptFileCount = [int]$missingScriptFiles.Count
    trackedScriptReferences = @($trackedScriptReferences.ToArray())
    untrackedScriptReferences = @($untrackedScriptReferences.ToArray())
    topUntrackedReferences = @($topUntrackedReferences)
    missingScriptFiles = @($missingScriptFiles.ToArray())
    ownerActionRequired = $true
    performsPublish = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    isPackageConsumerRuntimeProof = $false
    canPromoteRuntimeProof = $false
    boundary = "Release script reference index only. It scans tracked test source for PowerShell script references and reports tracking gaps; it is not runtime proof, not package-consumer proof, not publish approval, not release close approval, and not package push."
  }

  $jsonPath = Join-Path $OutputRoot "release-script-reference-index.json"
  $markdownPath = Join-Path $OutputRoot "release-script-reference-index.md"
  $record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

  $md = New-Object System.Collections.Generic.List[string]
  $md.Add("# Release Script Reference Index") | Out-Null
  $md.Add("") | Out-Null
  $md.Add("- indexState: ``$($record.indexState)``") | Out-Null
  $md.Add("- trackedTestFileCount: ``$($record.trackedTestFileCount)``") | Out-Null
  $md.Add("- uniqueScriptReferenceCount: ``$($record.uniqueScriptReferenceCount)``") | Out-Null
  $md.Add("- trackedScriptReferenceCount: ``$($record.trackedScriptReferenceCount)``") | Out-Null
  $md.Add("- untrackedScriptReferenceCount: ``$($record.untrackedScriptReferenceCount)``") | Out-Null
  $md.Add("- missingScriptFileCount: ``$($record.missingScriptFileCount)``") | Out-Null
  $md.Add("") | Out-Null
  $md.Add("## Top Untracked References") | Out-Null
  $md.Add("") | Out-Null
  $md.Add("| Script | Exists | Referencing Tests |") | Out-Null
  $md.Add("| --- | --- | ---: |") | Out-Null
  foreach ($entry in $topUntrackedReferences) {
    $md.Add("| ``$(ConvertTo-MarkdownCell $entry.scriptPath)`` | ``$($entry.exists)`` | ``$($entry.referencingTestCount)`` |") | Out-Null
  }
  $md.Add("") | Out-Null
  $md.Add("## Boundary") | Out-Null
  $md.Add("") | Out-Null
  $md.Add($record.boundary) | Out-Null
  [System.IO.File]::WriteAllText($markdownPath, (($md -join [Environment]::NewLine) + [Environment]::NewLine), $utf8)

  Write-Host "ReleaseScriptReferenceIndexState=$($record.indexState) Unique=$($record.uniqueScriptReferenceCount) Tracked=$($record.trackedScriptReferenceCount) Untracked=$($record.untrackedScriptReferenceCount) Missing=$($record.missingScriptFileCount)"
}
finally {
  Pop-Location
}
