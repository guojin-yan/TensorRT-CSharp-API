[CmdletBinding()]
param(
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

function Read-Json {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required JSON artifact was not found: $path"
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Get-ArrayOrEmpty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue @()
  if ($null -eq $value) {
    return @()
  }

  return @($value)
}

function Resolve-ClosureDecision {
  param([string]$ClosureBucket)

  switch ($ClosureBucket) {
    "already-safe-alternative-proof" { return "safe-alternative-documented-keep-deferred-history" }
    "docs-test-proof-needed" { return "docs-test-proof-ready-keep-deferred-history" }
    default { return "alias-proof-ready-keep-deferred-history" }
  }
}

function Convert-ToJsonArray {
  param([AllowNull()][object]$Values)

  $list = [System.Collections.Generic.List[string]]::new()
  foreach ($value in @($Values)) {
    if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
      $list.Add([string]$value)
    }
  }

  return ,$list
}

function Write-TextFileWithRetry {
  param(
    [string]$Path,
    [string]$Value,
    [System.Text.Encoding]$Encoding,
    [int]$RetryCount = 40,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $directory | Out-Null

  $tempFileName = ".{0}.{1}.tmp" -f ([System.IO.Path]::GetFileName($Path)), ([System.Guid]::NewGuid().ToString("N"))
  $tempPath = Join-Path $directory $tempFileName

  try {
    [System.IO.File]::WriteAllText($tempPath, $Value, $Encoding)

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
      try {
        Move-Item -LiteralPath $tempPath -Destination $Path -Force
        return
      }
      catch {
        if ($attempt -eq $RetryCount) {
          throw
        }

        Start-Sleep -Milliseconds $DelayMilliseconds
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
  }
}

$dashboard = Read-Json "artifacts\interface-coverage\deferred-btier-proof-closure-dashboard.json"
$selected = @(
  Get-ArrayOrEmpty -Object $dashboard -Name "selectedClosureCandidates" |
    Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "closureProofState" -DefaultValue "") -eq "alias-proof-ready" }
)

$closureRows = @(
  $selected | ForEach-Object {
    $safeAlternativeManifestIds = Get-ArrayOrEmpty -Object $_ -Name "safeAlternativeManifestIds"
    $deferredHistoryManifestIds = Get-ArrayOrEmpty -Object $_ -Name "deferredHistoryManifestIds"
    $sourceEvidence = Get-ArrayOrEmpty -Object $_ -Name "sourceEvidence"
    $closureBucket = [string](Get-PropertyOrDefault -Object $_ -Name "closureBucket" -DefaultValue "")

    [pscustomobject]@{
      interface = [string](Get-PropertyOrDefault -Object $_ -Name "interface" -DefaultValue "")
      version = [string](Get-PropertyOrDefault -Object $_ -Name "version" -DefaultValue "")
      class = [string](Get-PropertyOrDefault -Object $_ -Name "class" -DefaultValue "")
      method = [string](Get-PropertyOrDefault -Object $_ -Name "method" -DefaultValue "")
      tensorRtLine = [string](Get-PropertyOrDefault -Object $_ -Name "tensorRtLine" -DefaultValue "")
      ownershipRisk = [string](Get-PropertyOrDefault -Object $_ -Name "ownershipRisk" -DefaultValue "")
      safetyTier = [string](Get-PropertyOrDefault -Object $_ -Name "safetyTier" -DefaultValue "")
      designGroup = [string](Get-PropertyOrDefault -Object $_ -Name "designGroup" -DefaultValue "")
      closureBucket = $closureBucket
      closureProofState = [string](Get-PropertyOrDefault -Object $_ -Name "closureProofState" -DefaultValue "")
      evidenceKind = [string](Get-PropertyOrDefault -Object $_ -Name "evidenceKind" -DefaultValue "")
      safeAlternativeManifestIds = Convert-ToJsonArray -Values $safeAlternativeManifestIds
      deferredHistoryManifestIds = Convert-ToJsonArray -Values $deferredHistoryManifestIds
      sourceEvidence = Convert-ToJsonArray -Values $sourceEvidence
      publicApiExposurePolicy = [string](Get-PropertyOrDefault -Object $_ -Name "publicApiExposurePolicy" -DefaultValue "")
      recommendedAction = [string](Get-PropertyOrDefault -Object $_ -Name "recommendedAction" -DefaultValue "")
      closureDecision = Resolve-ClosureDecision -ClosureBucket $closureBucket
      closureEvidenceState = "safe-alternative-and-deferred-history-present"
      canDeleteDeferredRecord = $false
      canPromoteReleaseProof = $false
      isRuntimeExecutionProof = $false
      isPackageConsumerRuntimeProof = $false
    }
  }
)

$bucketSummaries = @(
  $closureRows |
    Group-Object closureBucket |
    Sort-Object Name |
    ForEach-Object {
      [pscustomobject]@{
        closureBucket = $_.Name
        candidateCount = $_.Count
        representativeInterfaces = @($_.Group | Select-Object -First 10 | ForEach-Object { "$($_.interface) [$($_.version)]" })
      }
    }
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "deferred-btier-alias-proof-closure-record"
  closureState = "alias-proof-ready-engineering-record"
  sourceDashboard = "artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.json"
  sourceArtifacts = @(
    "artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json"
  )
  dashboardSelectedCandidateCount = [int](Get-PropertyOrDefault -Object $dashboard -Name "selectedCandidateCount" -DefaultValue 0)
  dashboardSelectedAliasProofReadyCandidateCount = [int](Get-PropertyOrDefault -Object $dashboard -Name "selectedAliasProofReadyCandidateCount" -DefaultValue 0)
  closureCandidateCount = $closureRows.Count
  closureBuckets = $bucketSummaries
  closureCandidates = $closureRows
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canDeleteDeferredRecords = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  cAndDTierPolicy = "C/D tiers are excluded from alias proof closure and remain design-gated or keep-deferred."
  boundary = "B-tier alias proof closure record is engineering closure evidence only: it documents safe alternative manifests plus deferred history manifests and does not delete deferred records, publish packages, close release issues, or replace runtime proof."
  nextActions = @(
    "Use closureCandidates to add wrapper/docs/test proof for the safe public path.",
    "Keep deferred history manifests intact.",
    "Promote real APIs only when wrapper, smoke, and version guard proof exists.",
    "Do not treat this record as package-consumer-runtime proof."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\interface-coverage"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "deferred-btier-alias-proof-closure-record.json"
$markdownPath = Join-Path $artifactRoot "deferred-btier-alias-proof-closure-record.md"

$recordJson = $record | ConvertTo-Json -Depth 14
Write-TextFileWithRetry -Path $jsonPath -Value $recordJson -Encoding $utf8

$bucketRows = $bucketSummaries | ForEach-Object {
  "| ``$($_.closureBucket)`` | ``$($_.candidateCount)`` | $((@($_.representativeInterfaces) -join "<br>").Replace("|", "\|")) |"
}

$candidateRows = $closureRows | ForEach-Object {
  $safeAlternative = (@($_.safeAlternativeManifestIds) | Select-Object -First 4) -join "<br>"
  $deferredHistory = (@($_.deferredHistoryManifestIds) | Select-Object -First 4) -join "<br>"
  $evidence = (@($_.sourceEvidence) | Select-Object -First 4) -join "<br>"
  "| ``$($_.version)`` | ``$($_.interface)`` | ``$($_.closureBucket)`` | ``$($_.closureDecision)`` | $($safeAlternative.Replace("|", "\|")) | $($deferredHistory.Replace("|", "\|")) | $($evidence.Replace("|", "\|")) |"
}

$markdown = @"
# Deferred B-tier Alias Proof Closure Record

生成时间：$($record.generatedAtUtc)

## 总结

``recordKind=deferred-btier-alias-proof-closure-record``，``closureState=alias-proof-ready-engineering-record``。该记录只证明 selected B-tier 候选具备 safe alternative manifest 与 deferred history manifest 的工程闭环输入；它不是 release proof，不是 package-consumer-runtime proof，也不是删除 deferred 记录的许可。

| 项目 | 值 |
|---|---:|
| dashboard selected candidates | ``$($record.dashboardSelectedCandidateCount)`` |
| dashboard selected alias-proof-ready candidates | ``$($record.dashboardSelectedAliasProofReadyCandidateCount)`` |
| closure candidates | ``$($record.closureCandidateCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| canDeleteDeferredRecords | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isPackageConsumerRuntimeProof | ``False`` |

## Closure Buckets

| Bucket | Count | Representatives |
|---|---:|---|
$($bucketRows -join "`r`n")

## Closure Candidates

| Version | Interface | Bucket | Closure decision | Safe alternative manifests | Deferred history manifests | Source evidence |
|---|---|---|---|---|---|---|
$($candidateRows -join "`r`n")

## Boundary

$($record.boundary)

## C/D Policy

$($record.cAndDTierPolicy)
"@

Write-TextFileWithRetry -Path $markdownPath -Value $markdown -Encoding $utf8

Write-Output "Deferred B-tier alias proof closure record written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ClosureCandidateCount=$($record.closureCandidateCount)"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
Write-Output "CanDeleteDeferredRecords=False"
