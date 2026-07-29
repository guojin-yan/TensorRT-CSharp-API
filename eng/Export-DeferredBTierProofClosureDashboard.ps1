[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [int]$SelectedCandidateCount = 60
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

function Test-FileContains {
  param(
    [string]$RelativePath,
    [string]$Needle
  )

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $false
  }

  $content = Get-Content -LiteralPath $path -Raw -Encoding utf8
  return $content.IndexOf($Needle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Resolve-ClosureBucket {
  param([object]$Row)

  $className = [string](Get-PropertyOrDefault -Object $Row -Name "class" -DefaultValue "")
  $methodName = [string](Get-PropertyOrDefault -Object $Row -Name "method" -DefaultValue "")
  $designGroup = [string](Get-PropertyOrDefault -Object $Row -Name "designGroup" -DefaultValue "")
  $ownershipRisk = [string](Get-PropertyOrDefault -Object $Row -Name "ownershipRisk" -DefaultValue "")
  $implementationStatus = [string](Get-PropertyOrDefault -Object $Row -Name "implementationStatus" -DefaultValue "")
  $nativeManifestStatus = [string](Get-PropertyOrDefault -Object $Row -Name "nativeManifestStatus" -DefaultValue "")
  $nativeSourceStatus = [string](Get-PropertyOrDefault -Object $Row -Name "nativeSourceStatus" -DefaultValue "")
  $managedInteropStatus = [string](Get-PropertyOrDefault -Object $Row -Name "managedInteropStatus" -DefaultValue "")
  $reason = [string](Get-PropertyOrDefault -Object $Row -Name "reason" -DefaultValue "")
  $combined = "$className $methodName $designGroup $implementationStatus $nativeManifestStatus $nativeSourceStatus $managedInteropStatus $reason"

  if ($designGroup -match "plugin" -or $methodName -match "Plugin|Registry|Creator|Serialize|Deserialize|Refitter|Profiler|Monitor|Logger") {
    return "runtime-smoke-backfill-needed"
  }

  if ($ownershipRisk -ne "low" -or $methodName -match "create|set|add|clear|reset|report|load|register|deregister") {
    return "manual-review-required"
  }

  if ($methodName -match "Count|Name|Version|Namespace|Metadata|Field|Info|Snapshot|Values|Profiles|Bindings") {
    return "already-safe-alternative-proof"
  }

  if ($className -match "Builder|BuilderConfig|Engine|Runtime|Refitter|Parser|ExecutionContext|Network" -and
      $methodName -match "get|is|has|can|supports|query|read") {
    return "docs-test-proof-needed"
  }

  if ($combined -match "count|copy|presence|probe|snapshot|metadata") {
    return "already-safe-alternative-proof"
  }

  if ($implementationStatus -eq "implemented-with-deferred-history" -and
      $nativeManifestStatus -eq "present" -and
      $nativeSourceStatus -eq "present" -and
      $managedInteropStatus -eq "generated-or-manual") {
    return "alias-closure-needed"
  }

  return "manual-review-required"
}

function Resolve-EvidenceKind {
  param(
    [object]$Row,
    [string]$ClosureBucket
  )

  switch ($ClosureBucket) {
    "already-safe-alternative-proof" { return "safe-alternative-present" }
    "alias-closure-needed" { return "manifest-source-managed-alias-proof" }
    "docs-test-proof-needed" { return "docs-and-quality-gate-proof" }
    "runtime-smoke-backfill-needed" { return "smoke-backfill-planning-input" }
    default { return "manual-review-planning-input" }
  }
}

function Resolve-EvidenceFiles {
  param([object]$Row)

  $className = [string](Get-PropertyOrDefault -Object $Row -Name "class" -DefaultValue "")
  $methodName = [string](Get-PropertyOrDefault -Object $Row -Name "method" -DefaultValue "")
  $tensorRtLine = [string](Get-PropertyOrDefault -Object $Row -Name "tensorRtLine" -DefaultValue "")
  $designGroup = [string](Get-PropertyOrDefault -Object $Row -Name "designGroup" -DefaultValue "")
  $evidence = New-Object System.Collections.Generic.List[string]

  $evidence.Add("artifacts/interface-coverage/deferred-candidate-safety-triage.json")
  $evidence.Add("artifacts/interface-coverage/tensorrt-interface-comparison.csv")

  switch ($tensorRtLine) {
    "8" { $evidence.Add("native/manifests/tensorrt/v8") }
    "10" { $evidence.Add("native/manifests/tensorrt/v10") }
    "11" { $evidence.Add("native/manifests/tensorrt/v11") }
    default { $evidence.Add("native/manifests/tensorrt") }
  }

  $evidence.Add("native/src/tensorrt")
  $evidence.Add("src/JYPPX.TensorRtSharp")
  $evidence.Add("tests/JYPPX.ProjectQuality.Tests")

  if ($designGroup -match "plugin") {
    $evidence.Add("src/JYPPX.TensorRtSharp/Plugins/TensorRtPluginRegistryInventory.cs")
    $evidence.Add("smoke/PluginRegistryInventorySmokeRunner")
  }
  elseif ($className -match "Builder|BuilderConfig") {
    $evidence.Add("src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.cs")
    $evidence.Add("src/JYPPX.TensorRtSharp/Builder/TensorRtBuilderConfig.cs")
  }
  elseif ($className -match "Runtime") {
    $evidence.Add("src/JYPPX.TensorRtSharp/Runtime/TensorRtRuntime.cs")
  }
  elseif ($className -match "Engine") {
    $evidence.Add("src/JYPPX.TensorRtSharp/Engine")
  }
  elseif ($className -match "ExecutionContext") {
    $evidence.Add("src/JYPPX.TensorRtSharp/Execution/TensorRtExecutionContext.cs")
  }
  elseif ($className -match "Parser") {
    $evidence.Add("src/JYPPX.TensorRtSharp/Parsing")
    $evidence.Add("samples/OnnxToEngine")
  }

  if (-not [string]::IsNullOrWhiteSpace($methodName) -and (Test-FileContains -RelativePath "src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.cs" -Needle $methodName)) {
    $evidence.Add("src/JYPPX.TensorRtSharp/Builder/TensorRtBuilder.cs#$methodName")
  }

  return @($evidence | Select-Object -Unique)
}

function Resolve-PublicApiPolicy {
  param([string]$ClosureBucket)

  switch ($ClosureBucket) {
    "already-safe-alternative-proof" { return "prefer-existing-safe-wrapper-do-not-expose-deferred-pointer" }
    "alias-closure-needed" { return "close-by-alias-or-proof-keep-deferred-history" }
    "docs-test-proof-needed" { return "require-docs-and-quality-gate-before-public-promotion" }
    "runtime-smoke-backfill-needed" { return "require-smoke-backfill-before-public-promotion" }
    default { return "manual-review-before-public-api" }
  }
}

function Resolve-RecommendedAction {
  param([string]$ClosureBucket)

  switch ($ClosureBucket) {
    "already-safe-alternative-proof" { return "document existing safe alternative and add proof test; keep deferred history intact." }
    "alias-closure-needed" { return "close alias/proof gap across manifest, source, managed wrapper, docs, and tests without deleting deferred records." }
    "docs-test-proof-needed" { return "add docs and ProjectQuality proof that the safe surface is covered by existing interop/wrapper." }
    "runtime-smoke-backfill-needed" { return "add or extend smoke coverage before considering public promotion; do not claim runtime proof." }
    default { return "hold for manual review and keep deferred." }
  }
}

function Split-ManifestIds {
  param([string]$MatchedManifestIds)

  if ([string]::IsNullOrWhiteSpace($MatchedManifestIds)) {
    return @()
  }

  return @(
    $MatchedManifestIds.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries) |
      ForEach-Object { $_.Trim() } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  )
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

function Resolve-ClosureProofState {
  param(
    [string]$ClosureBucket,
    [string[]]$SafeAlternativeManifestIds,
    [string[]]$DeferredHistoryManifestIds
  )

  if ($ClosureBucket -eq "runtime-smoke-backfill-needed") {
    return "smoke-backfill-required"
  }

  if ($ClosureBucket -eq "manual-review-required") {
    return "manual-review-required"
  }

  if ($SafeAlternativeManifestIds.Count -gt 0 -and $DeferredHistoryManifestIds.Count -gt 0) {
    return "alias-proof-ready"
  }

  return "planning-input-only"
}

$triage = Read-Json "artifacts\interface-coverage\deferred-candidate-safety-triage.json"
$rows = @($triage.rows)
$tierSummaries = @($triage.tierSummaries)
$bRows = @($rows | Where-Object { $_.safetyTier -eq "B - safe-alternative-or-alias" })
$bUniqueRows = @(
  $bRows |
    Group-Object tensorRtLine, interface, matchedManifestIds |
    ForEach-Object { $_.Group[0] }
)
$cRows = @($rows | Where-Object { $_.safetyTier -eq "C - design-gate-required" })
$dRows = @($rows | Where-Object { $_.safetyTier -eq "D - keep-deferred" })
$aRows = @($rows | Where-Object { $_.safetyTier -eq "A - immediate-safe" })

$closureRows = @(
  $bUniqueRows | ForEach-Object {
    $bucket = Resolve-ClosureBucket -Row $_
    $evidenceKind = Resolve-EvidenceKind -Row $_ -ClosureBucket $bucket
    $tensorRtLine = [string](Get-PropertyOrDefault -Object $_ -Name "tensorRtLine" -DefaultValue "")
    $matchedManifestIds = [string](Get-PropertyOrDefault -Object $_ -Name "matchedManifestIds" -DefaultValue "")
    $manifestIds = Split-ManifestIds -MatchedManifestIds $matchedManifestIds
    $safeAlternativeManifestIds = @($manifestIds | Where-Object { $_ -notmatch "deferred" })
    $deferredHistoryManifestIds = @($manifestIds | Where-Object { $_ -match "deferred" })
    $closureProofState = Resolve-ClosureProofState `
      -ClosureBucket $bucket `
      -SafeAlternativeManifestIds $safeAlternativeManifestIds `
      -DeferredHistoryManifestIds $deferredHistoryManifestIds
    [pscustomobject]@{
      interface = [string](Get-PropertyOrDefault -Object $_ -Name "interface" -DefaultValue "")
      class = [string](Get-PropertyOrDefault -Object $_ -Name "class" -DefaultValue "")
      method = [string](Get-PropertyOrDefault -Object $_ -Name "method" -DefaultValue "")
      version = "TRT$tensorRtLine"
      tensorRtLine = $tensorRtLine
      header = [string](Get-PropertyOrDefault -Object $_ -Name "header" -DefaultValue "")
      implementationStatus = [string](Get-PropertyOrDefault -Object $_ -Name "implementationStatus" -DefaultValue "")
      nativeManifestStatus = [string](Get-PropertyOrDefault -Object $_ -Name "nativeManifestStatus" -DefaultValue "")
      nativeSourceStatus = [string](Get-PropertyOrDefault -Object $_ -Name "nativeSourceStatus" -DefaultValue "")
      managedInteropStatus = [string](Get-PropertyOrDefault -Object $_ -Name "managedInteropStatus" -DefaultValue "")
      matchedManifestIds = $matchedManifestIds
      safeAlternativeManifestIds = $safeAlternativeManifestIds
      deferredHistoryManifestIds = $deferredHistoryManifestIds
      ownershipRisk = [string](Get-PropertyOrDefault -Object $_ -Name "ownershipRisk" -DefaultValue "")
      safetyTier = [string](Get-PropertyOrDefault -Object $_ -Name "safetyTier" -DefaultValue "")
      designGroup = [string](Get-PropertyOrDefault -Object $_ -Name "designGroup" -DefaultValue "")
      closureBucket = $bucket
      evidenceKind = $evidenceKind
      closureProofState = $closureProofState
      sourceEvidence = Resolve-EvidenceFiles -Row $_
      recommendedAction = Resolve-RecommendedAction -ClosureBucket $bucket
      publicApiExposurePolicy = Resolve-PublicApiPolicy -ClosureBucket $bucket
      canDeleteDeferredRecord = $false
      canPromoteReleaseProof = $false
      reason = [string](Get-PropertyOrDefault -Object $_ -Name "reason" -DefaultValue "")
    }
  }
)

$bucketOrder = @(
  "already-safe-alternative-proof",
  "alias-closure-needed",
  "docs-test-proof-needed",
  "runtime-smoke-backfill-needed",
  "manual-review-required"
)

$bucketSummaries = @(
  foreach ($bucket in $bucketOrder) {
    $items = @($closureRows | Where-Object { $_.closureBucket -eq $bucket })
    [pscustomobject]@{
      closureBucket = $bucket
      candidateCount = $items.Count
      aliasProofReadyCount = @($items | Where-Object { $_.closureProofState -eq "alias-proof-ready" }).Count
      representativeInterfaces = @($items | Select-Object -First 10 | ForEach-Object { "$($_.interface) [$($_.version)]" })
    }
  }
)

$selectedCandidates = @(
  foreach ($bucket in $bucketOrder) {
    $closureRows |
      Where-Object { $_.closureBucket -eq $bucket -and $_.ownershipRisk -eq "low" } |
      Sort-Object @{ Expression = "tensorRtLine"; Descending = $true }, "class", "method", "interface" |
      Select-Object -First ([Math]::Max(1, [Math]::Ceiling($SelectedCandidateCount / $bucketOrder.Count)))
  }
) | Select-Object -First $SelectedCandidateCount

if ($selectedCandidates.Count -lt $SelectedCandidateCount) {
  $selectedKeys = @{}
  foreach ($candidate in $selectedCandidates) {
    $selectedKeys["$($candidate.version)|$($candidate.interface)|$($candidate.matchedManifestIds)"] = $true
  }

  $fill = $closureRows |
    Where-Object { -not $selectedKeys.ContainsKey("$($_.version)|$($_.interface)|$($_.matchedManifestIds)") } |
    Sort-Object "closureBucket", @{ Expression = "tensorRtLine"; Descending = $true }, "class", "method" |
    Select-Object -First ($SelectedCandidateCount - $selectedCandidates.Count)
  $selectedCandidates = @($selectedCandidates + $fill)
}

$excludedTierSummaries = @(
  [pscustomobject]@{
    safetyTier = "A - immediate-safe"
    count = $aRows.Count
    closurePolicy = "not part of B-tier proof closure dashboard"
  }
  [pscustomobject]@{
    safetyTier = "C - design-gate-required"
    count = $cRows.Count
    closurePolicy = "excluded from B-tier closure; requires design gate"
  }
  [pscustomobject]@{
    safetyTier = "D - keep-deferred"
    count = $dRows.Count
    closurePolicy = "excluded from B-tier closure; must remain deferred"
  }
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "deferred-btier-proof-closure-dashboard"
  closureState = "planning-input-only"
  sourceTriageKind = [string](Get-PropertyOrDefault -Object $triage -Name "triageKind" -DefaultValue "")
  sourceMatrix = [string](Get-PropertyOrDefault -Object $triage -Name "sourceMatrix" -DefaultValue "")
  sourceArtifacts = @(
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.md",
    "artifacts/interface-coverage/tensorrt-interface-comparison.csv"
  )
  totalTriageRowCount = [int](Get-PropertyOrDefault -Object $triage -Name "totalTriageRowCount" -DefaultValue 0)
  totalBTierCount = $bRows.Count
  uniqueBTierCandidateCount = $bUniqueRows.Count
  totalClosureCandidateCount = $closureRows.Count
  selectedCandidateCount = $selectedCandidates.Count
  selectedCandidateTargetCount = $SelectedCandidateCount
  aliasProofReadyCandidateCount = @($closureRows | Where-Object { $_.closureProofState -eq "alias-proof-ready" }).Count
  selectedAliasProofReadyCandidateCount = @($selectedCandidates | Where-Object { $_.closureProofState -eq "alias-proof-ready" }).Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  canDeleteDeferredRecords = $false
  bTierSafetyTier = "B - safe-alternative-or-alias"
  bTierRecommendedAction = "alias-or-proof-safe-alternative"
  closureBuckets = $bucketSummaries
  selectedClosureCandidates = $selectedCandidates
  closureCandidates = $closureRows
  excludedTierSummaries = $excludedTierSummaries
  nonSubstituteProofKinds = @(
    "B-tier proof closure dashboard",
    "safe-alternative planning input",
    "alias closure planning input",
    "docs/test proof planning input",
    "runtime smoke backfill planning input",
    "manual-review planning input",
    "deferred safety triage"
  )
  boundary = "B-tier proof closure dashboard is engineering closure input only. It is not release proof, package-consumer-runtime proof, owner approval, post-publish verification, or permission to delete deferred records."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\interface-coverage"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "deferred-btier-proof-closure-dashboard.json"
$markdownPath = Join-Path $artifactRoot "deferred-btier-proof-closure-dashboard.md"

$recordJson = $record | ConvertTo-Json -Depth 14
Write-TextFileWithRetry -Path $jsonPath -Value $recordJson -Encoding $utf8

$bucketRows = $bucketSummaries | ForEach-Object {
  "| ``$($_.closureBucket)`` | ``$($_.candidateCount)`` | ``$($_.aliasProofReadyCount)`` | $((@($_.representativeInterfaces) -join "<br>").Replace("|", "\|")) |"
}

$candidateRows = $selectedCandidates | ForEach-Object {
  $evidence = (@($_.sourceEvidence) | Select-Object -First 4) -join "<br>"
  $safeAlternative = (@($_.safeAlternativeManifestIds) | Select-Object -First 3) -join "<br>"
  $deferredHistory = (@($_.deferredHistoryManifestIds) | Select-Object -First 3) -join "<br>"
  "| ``$($_.version)`` | ``$($_.interface)`` | ``$($_.closureBucket)`` | ``$($_.closureProofState)`` | ``$($_.evidenceKind)`` | $($safeAlternative.Replace("|", "\|")) | $($deferredHistory.Replace("|", "\|")) | $($evidence.Replace("|", "\|")) | $($_.recommendedAction.Replace("|", "\|")) |"
}

$excludedRows = $excludedTierSummaries | ForEach-Object {
  "| ``$($_.safetyTier)`` | ``$($_.count)`` | $($_.closurePolicy.Replace("|", "\|")) |"
}

$markdown = @"
# Deferred B-tier Proof Closure Dashboard

生成时间：$($record.generatedAtUtc)

## 总结

``recordKind=deferred-btier-proof-closure-dashboard``，``closureState=planning-input-only``。该产物只用于 B 类 deferred 候选的工程收口规划，不是 release proof，不是 package-consumer-runtime proof，也不是删除 deferred 记录的许可。

| 项目 | 值 |
|---|---|
| total triage rows | ``$($record.totalTriageRowCount)`` |
| total B-tier rows | ``$($record.totalBTierCount)`` |
| unique B-tier closure candidates | ``$($record.uniqueBTierCandidateCount)`` |
| selected closure candidates | ``$($record.selectedCandidateCount)`` |
| alias-proof-ready candidates | ``$($record.aliasProofReadyCandidateCount)`` |
| selected alias-proof-ready candidates | ``$($record.selectedAliasProofReadyCandidateCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isPackageConsumerRuntimeProof | ``False`` |
| canDeleteDeferredRecords | ``False`` |

## Closure Buckets

| Bucket | Count | Alias-proof-ready | Representatives |
|---|---:|---:|---|
$($bucketRows -join "`r`n")

## Selected Closure Candidates

| Version | Interface | Bucket | Proof state | Evidence kind | Safe alternative manifests | Deferred history manifests | Evidence | Recommended action |
|---|---|---|---|---|---|---|---|---|
$($candidateRows -join "`r`n")

## Excluded Tiers

| Safety tier | Count | Policy |
|---|---:|---|
$($excludedRows -join "`r`n")

## Boundary

$($record.boundary)

## Next Batch Rule

下一批优先从 ``alias-closure-needed`` 与 ``docs-test-proof-needed`` 中选择低 ownership risk、只读、查询型接口；C/D 不进入本 dashboard 的 closure batch。
"@

Write-TextFileWithRetry -Path $markdownPath -Value $markdown -Encoding $utf8

Write-Output "Deferred B-tier proof closure dashboard written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "TotalBTierCount=$($record.totalBTierCount)"
Write-Output "UniqueBTierCandidateCount=$($record.uniqueBTierCandidateCount)"
Write-Output "TotalClosureCandidateCount=$($record.totalClosureCandidateCount)"
Write-Output "SelectedCandidateCount=$($record.selectedCandidateCount)"
Write-Output "AliasProofReadyCandidateCount=$($record.aliasProofReadyCandidateCount)"
Write-Output "SelectedAliasProofReadyCandidateCount=$($record.selectedAliasProofReadyCandidateCount)"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
Write-Output "CanDeleteDeferredRecords=False"
