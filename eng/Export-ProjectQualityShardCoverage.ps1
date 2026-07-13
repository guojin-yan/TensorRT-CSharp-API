[CmdletBinding()]
param(
  [string]$InventoryPath = "",
  [string]$ShardRoot = "",
  [string]$OutputDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepositoryPath {
  param(
    [AllowEmptyString()][string]$Path,
    [Parameter(Mandatory = $true)][string]$DefaultRelativePath
  )

  $candidate = if ([string]::IsNullOrWhiteSpace($Path)) {
    Join-Path $repositoryRoot $DefaultRelativePath
  }
  elseif ([IO.Path]::IsPathRooted($Path)) {
    $Path
  }
  else {
    Join-Path $repositoryRoot $Path
  }

  return [IO.Path]::GetFullPath($candidate)
}

function ConvertTo-RelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  return [IO.Path]::GetRelativePath($repositoryRoot, [IO.Path]::GetFullPath($Path)).Replace("\", "/")
}

function Get-OptionalProperty {
  param(
    [Parameter(Mandatory = $true)][object]$Object,
    [Parameter(Mandatory = $true)][string]$Name,
    [object]$DefaultValue = $null
  )

  $property = $Object.PSObject.Properties[$Name]
  if ($null -eq $property) {
    return $DefaultValue
  }

  return $property.Value
}

$InventoryPath = Resolve-RepositoryPath -Path $InventoryPath -DefaultRelativePath "artifacts\test-analysis\project-quality-test-inventory.json"
$ShardRoot = Resolve-RepositoryPath -Path $ShardRoot -DefaultRelativePath "artifacts\test-analysis\project-quality-shards"
$OutputDirectory = Resolve-RepositoryPath -Path $OutputDirectory -DefaultRelativePath "artifacts\test-analysis"

if (-not (Test-Path -LiteralPath $InventoryPath -PathType Leaf)) {
  throw "ProjectQuality inventory not found: $InventoryPath"
}
if (-not (Test-Path -LiteralPath $ShardRoot -PathType Container)) {
  throw "ProjectQuality shard root not found: $ShardRoot"
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$inventory = Get-Content -LiteralPath $InventoryPath -Raw | ConvertFrom-Json
$inventoryHash = (Get-FileHash -LiteralPath $InventoryPath -Algorithm SHA256).Hash.ToLowerInvariant()
$inventoryClasses = @(
  $inventory.shards |
    ForEach-Object { $_.classes } |
    ForEach-Object { [string]$_ } |
    Sort-Object -Unique
)
$coveredClasses = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$evidenceUnits = [Collections.Generic.List[object]]::new()
$invalidEvidence = [Collections.Generic.List[object]]::new()
$summaryFiles = @(Get-ChildItem -LiteralPath $ShardRoot -Recurse -File -Filter "summary.json" | Sort-Object FullName)
$passingExecutionUnitCount = 0

foreach ($summaryFile in $summaryFiles) {
  try {
    $summary = Get-Content -LiteralPath $summaryFile.FullName -Raw | ConvertFrom-Json
  }
  catch {
    $invalidEvidence.Add([pscustomobject][ordered]@{
        summaryPath = ConvertTo-RelativePath $summaryFile.FullName
        unitId = ""
        reason = "summary-json-invalid"
        detail = $_.Exception.Message
      })
    continue
  }

  $runId = [string](Get-OptionalProperty -Object $summary -Name "runId" -DefaultValue $summaryFile.Directory.Name)
  foreach ($result in @($summary.results)) {
    if ([string](Get-OptionalProperty -Object $result -Name "state" -DefaultValue "") -ne "passed") {
      continue
    }

    $passingExecutionUnitCount++
    $unitId = [string](Get-OptionalProperty -Object $result -Name "id" -DefaultValue "")
    $trxRelativePath = [string](Get-OptionalProperty -Object $result -Name "trxPath" -DefaultValue "")
    $expectedTrxHash = ([string](Get-OptionalProperty -Object $result -Name "trxSha256" -DefaultValue "")).ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($trxRelativePath) -or [string]::IsNullOrWhiteSpace($expectedTrxHash)) {
      $invalidEvidence.Add([pscustomobject][ordered]@{
          summaryPath = ConvertTo-RelativePath $summaryFile.FullName
          unitId = $unitId
          reason = "trx-proof-metadata-missing"
          detail = "A passed execution unit must record trxPath and trxSha256."
        })
      continue
    }

    $trxPath = if ([IO.Path]::IsPathRooted($trxRelativePath)) {
      [IO.Path]::GetFullPath($trxRelativePath)
    }
    else {
      [IO.Path]::GetFullPath((Join-Path $repositoryRoot ($trxRelativePath.Replace("/", "\"))))
    }

    if (-not (Test-Path -LiteralPath $trxPath -PathType Leaf)) {
      $invalidEvidence.Add([pscustomobject][ordered]@{
          summaryPath = ConvertTo-RelativePath $summaryFile.FullName
          unitId = $unitId
          reason = "trx-file-missing"
          detail = $trxRelativePath
        })
      continue
    }

    $actualTrxHash = (Get-FileHash -LiteralPath $trxPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualTrxHash -ne $expectedTrxHash) {
      $invalidEvidence.Add([pscustomobject][ordered]@{
          summaryPath = ConvertTo-RelativePath $summaryFile.FullName
          unitId = $unitId
          reason = "trx-sha256-mismatch"
          detail = "expected=$expectedTrxHash actual=$actualTrxHash"
        })
      continue
    }

    try {
      [xml]$trx = Get-Content -LiteralPath $trxPath -Raw
      $trxClasses = @(
        $trx.GetElementsByTagName("TestMethod") |
          ForEach-Object { [string]$_.className } |
          Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
          Sort-Object -Unique
      )
    }
    catch {
      $invalidEvidence.Add([pscustomobject][ordered]@{
          summaryPath = ConvertTo-RelativePath $summaryFile.FullName
          unitId = $unitId
          reason = "trx-xml-invalid"
          detail = $_.Exception.Message
        })
      continue
    }

    if ($trxClasses.Count -eq 0) {
      $invalidEvidence.Add([pscustomobject][ordered]@{
          summaryPath = ConvertTo-RelativePath $summaryFile.FullName
          unitId = $unitId
          reason = "trx-class-list-empty"
          detail = $trxRelativePath
        })
      continue
    }

    foreach ($className in $trxClasses) {
      [void]$coveredClasses.Add($className)
    }

    $counters = Get-OptionalProperty -Object $result -Name "counters" -DefaultValue $null
    $evidenceUnits.Add([pscustomobject][ordered]@{
        runId = $runId
        unitId = $unitId
        parentShardId = [string](Get-OptionalProperty -Object $result -Name "parentShardId" -DefaultValue "")
        classCount = $trxClasses.Count
        classNames = @($trxClasses)
        passedTestCount = if ($null -eq $counters) { 0 } else { [int](Get-OptionalProperty -Object $counters -Name "passed" -DefaultValue 0) }
        durationSeconds = [double](Get-OptionalProperty -Object $result -Name "durationSeconds" -DefaultValue 0)
        trxPath = ConvertTo-RelativePath $trxPath
        trxSha256 = $actualTrxHash
        summaryPath = ConvertTo-RelativePath $summaryFile.FullName
      })
  }
}

$missingClasses = @($inventoryClasses | Where-Object { -not $coveredClasses.Contains($_) })
$shardCoverage = @(
  foreach ($shard in $inventory.shards) {
    $classes = @($shard.classes | ForEach-Object { [string]$_ })
    $covered = @($classes | Where-Object { $coveredClasses.Contains($_) })
    $missing = @($classes | Where-Object { -not $coveredClasses.Contains($_) })
    [pscustomobject][ordered]@{
      id = [string]$shard.id
      inventoryClassCount = $classes.Count
      coveredClassCount = $covered.Count
      missingClassCount = $missing.Count
      coveragePercent = if ($classes.Count -eq 0) { 0 } else { [Math]::Round(($covered.Count * 100.0) / $classes.Count, 2) }
      missingClasses = @($missing)
    }
  }
)

$allClassesCovered = $missingClasses.Count -eq 0
$coverageState = if ($allClassesCovered -and $invalidEvidence.Count -eq 0) {
  "complete-class-coverage"
}
else {
  "incomplete-class-coverage"
}

$record = [pscustomobject][ordered]@{
  recordKind = "project-quality-shard-class-coverage"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  coverageState = $coverageState
  inventoryPath = ConvertTo-RelativePath $InventoryPath
  inventorySha256 = $inventoryHash
  inventoryTestCount = [int]$inventory.testCount
  inventoryClassCount = $inventoryClasses.Count
  coveredClassCount = $coveredClasses.Count
  missingClassCount = $missingClasses.Count
  allClassesCovered = $allClassesCovered
  sourceSummaryCount = $summaryFiles.Count
  passingExecutionUnitCount = $passingExecutionUnitCount
  strictPassedTrxCount = $evidenceUnits.Count
  invalidEvidenceCount = $invalidEvidence.Count
  shardCoverage = @($shardCoverage)
  missingClasses = @($missingClasses)
  invalidEvidence = @($invalidEvidence)
  evidenceUnits = @($evidenceUnits)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "This is cumulative class-level ProjectQuality coverage from hash-verified passed TRX files. It is not a one-shot whole-suite run, package-consumer runtime proof, publish approval, or release-close proof."
}

$jsonPath = Join-Path $OutputDirectory "project-quality-shard-class-coverage.json"
$markdownPath = Join-Path $OutputDirectory "project-quality-shard-class-coverage.md"
$record | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = [Collections.Generic.List[string]]::new()
$markdown.Add("# ProjectQuality Shard Class Coverage")
$markdown.Add("")
$markdown.Add("- Coverage state: ``$coverageState``")
$markdown.Add("- Inventory: ``$($record.inventoryTestCount)`` tests / ``$($record.inventoryClassCount)`` classes")
$markdown.Add("- Covered classes: ``$($record.coveredClassCount)``")
$markdown.Add("- Missing classes: ``$($record.missingClassCount)``")
$markdown.Add("- Hash-verified passed TRX files: ``$($record.strictPassedTrxCount)``")
$markdown.Add("- Invalid evidence records: ``$($record.invalidEvidenceCount)``")
$markdown.Add("- Performs publish: ``False``")
$markdown.Add("- Can publish publicly: ``False``")
$markdown.Add("- Can close release issue: ``False``")
$markdown.Add("")
$markdown.Add("## Boundary")
$markdown.Add("")
$markdown.Add($record.boundary)
$markdown.Add("")
$markdown.Add("## Shards")
$markdown.Add("")
$markdown.Add("| Shard | Inventory classes | Covered | Missing | Coverage |")
$markdown.Add("| --- | ---: | ---: | ---: | ---: |")
foreach ($shard in $shardCoverage) {
  $markdown.Add("| $($shard.id) | $($shard.inventoryClassCount) | $($shard.coveredClassCount) | $($shard.missingClassCount) | $($shard.coveragePercent)% |")
}
$markdown.Add("")
$markdown.Add("## Evidence")
$markdown.Add("")
$markdown.Add("Each accepted execution unit has ``state=passed``, an existing TRX file, and a matching recorded SHA256. Failed, timed-out, preview-only, missing, or hash-mismatched records do not contribute coverage.")
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ProjectQuality shard coverage written."
Write-Host "JSON=$(ConvertTo-RelativePath $jsonPath)"
Write-Host "Markdown=$(ConvertTo-RelativePath $markdownPath)"
Write-Host "CoverageState=$coverageState Classes=$($record.coveredClassCount)/$($record.inventoryClassCount) Missing=$($record.missingClassCount) ValidTrx=$($record.strictPassedTrxCount) InvalidEvidence=$($record.invalidEvidenceCount)"
