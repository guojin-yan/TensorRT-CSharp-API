[CmdletBinding()]
param(
  [string[]]$Shard = @("A-F", "G-M", "N-S", "T-Z"),
  [ValidateRange(30, 7200)]
  [int]$TimeoutSeconds = 600,
  [string]$Configuration = "Debug",
  [string]$TestProject = "",
  [string]$InventoryPath = "",
  [string]$CoveragePath = "",
  [string]$OutputRoot = "",
  [string]$RunId = "",
  [ValidateRange(0, 1000)]
  [int]$BatchSize = 0,
  [int[]]$Batch = @(),
  [string]$ClassNamePattern = "",
  [switch]$MissingOnly,
  [ValidateRange(1, 100)]
  [int]$DurationRankingCount = 20,
  [switch]$RefreshInventory,
  [switch]$PreviewOnly,
  [switch]$ContinueOnFailure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($TestProject)) {
  $TestProject = Join-Path $repositoryRoot "tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj"
}
elseif (-not [IO.Path]::IsPathRooted($TestProject)) {
  $TestProject = Join-Path $repositoryRoot $TestProject
}

if ([string]::IsNullOrWhiteSpace($InventoryPath)) {
  $InventoryPath = Join-Path $repositoryRoot "artifacts\test-analysis\project-quality-test-inventory.json"
}
elseif (-not [IO.Path]::IsPathRooted($InventoryPath)) {
  $InventoryPath = Join-Path $repositoryRoot $InventoryPath
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $repositoryRoot "artifacts\test-analysis\project-quality-shards"
}
elseif (-not [IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $repositoryRoot $OutputRoot
}

if ($MissingOnly) {
  if ([string]::IsNullOrWhiteSpace($CoveragePath)) {
    $CoveragePath = Join-Path $repositoryRoot "artifacts\test-analysis\project-quality-shard-class-coverage.json"
  }
  elseif (-not [IO.Path]::IsPathRooted($CoveragePath)) {
    $CoveragePath = Join-Path $repositoryRoot $CoveragePath
  }
}

if ($RefreshInventory -or -not (Test-Path -LiteralPath $InventoryPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-ProjectQualityTestInventory.ps1") `
    -Configuration $Configuration `
    -TestProject $TestProject `
    -OutputDirectory (Split-Path -Parent $InventoryPath)
}

if (-not (Test-Path -LiteralPath $InventoryPath -PathType Leaf)) {
  throw "ProjectQuality inventory not found: $InventoryPath"
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
  $RunId = [DateTime]::Now.ToString("yyyyMMdd-HHmmss")
}

function ConvertTo-RelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  return [IO.Path]::GetRelativePath($repositoryRoot, [IO.Path]::GetFullPath($Path)).Replace("\", "/")
}

function Stop-ProcessTree {
  param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)

  $attempted = $false
  $exited = $Process.HasExited
  $errorMessage = ""
  if (-not $exited) {
    $attempted = $true
    try {
      $Process.Kill($true)
      $exited = $Process.WaitForExit(30000)
    }
    catch {
      $errorMessage = $_.Exception.Message
      try {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        $exited = $Process.WaitForExit(10000)
      }
      catch {
        if ([string]::IsNullOrWhiteSpace($errorMessage)) {
          $errorMessage = $_.Exception.Message
        }
      }
    }
  }

  return [pscustomobject][ordered]@{
    attempted = $attempted
    processExited = $exited
    error = $errorMessage
  }
}

function Get-TrxCounters {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    return [pscustomobject][ordered]@{
      available = $false
      total = 0
      executed = 0
      passed = 0
      failed = 0
      error = 0
      timeout = 0
      aborted = 0
      inconclusive = 0
      notExecuted = 0
    }
  }

  [xml]$document = Get-Content -LiteralPath $Path -Raw
  $counters = $document.TestRun.ResultSummary.Counters
  if ($null -eq $counters) {
    return [pscustomobject][ordered]@{
      available = $false
      total = 0
      executed = 0
      passed = 0
      failed = 0
      error = 0
      timeout = 0
      aborted = 0
      inconclusive = 0
      notExecuted = 0
    }
  }

  function Read-Counter {
    param([object]$Counters, [string]$Name)
    $property = $Counters.PSObject.Properties[$Name]
    if ($null -eq $property -or [string]::IsNullOrWhiteSpace([string]$property.Value)) {
      return 0
    }
    return [int]$property.Value
  }

  return [pscustomobject][ordered]@{
    available = $true
    total = Read-Counter $counters "total"
    executed = Read-Counter $counters "executed"
    passed = Read-Counter $counters "passed"
    failed = Read-Counter $counters "failed"
    error = Read-Counter $counters "error"
    timeout = Read-Counter $counters "timeout"
    aborted = Read-Counter $counters "aborted"
    inconclusive = Read-Counter $counters "inconclusive"
    notExecuted = Read-Counter $counters "notExecuted"
  }
}

$inventory = Get-Content -LiteralPath $InventoryPath -Raw | ConvertFrom-Json
$inventorySha256 = (Get-FileHash -LiteralPath $InventoryPath -Algorithm SHA256).Hash.ToLowerInvariant()
$coverage = $null
$coverageSha256 = ""
$missingClassSet = $null
$requestedMissingClasses = @()
if ($MissingOnly) {
  if (-not (Test-Path -LiteralPath $CoveragePath -PathType Leaf)) {
    throw "ProjectQuality shard coverage not found: $CoveragePath"
  }

  $coverage = Get-Content -LiteralPath $CoveragePath -Raw | ConvertFrom-Json
  if ([string]$coverage.recordKind -ne "project-quality-shard-class-coverage") {
    throw "Unexpected ProjectQuality coverage record kind in '$CoveragePath'."
  }
  if (([string]$coverage.inventorySha256).ToLowerInvariant() -ne $inventorySha256) {
    throw "ProjectQuality coverage inventory SHA256 does not match '$InventoryPath'. Refresh shard coverage before using -MissingOnly."
  }

  $coverageSha256 = (Get-FileHash -LiteralPath $CoveragePath -Algorithm SHA256).Hash.ToLowerInvariant()
  $requestedMissingClasses = @($coverage.missingClasses | ForEach-Object { [string]$_ } | Sort-Object -Unique)
  $missingClassSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
  foreach ($className in $requestedMissingClasses) {
    [void]$missingClassSet.Add($className)
  }
}
$knownShards = @($inventory.shards | ForEach-Object id)
$requestedShards = @(
  $Shard |
    ForEach-Object { $_ -split "," } |
    ForEach-Object { $_.Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -Unique
)
if ($requestedShards.Count -eq 0) {
  throw "At least one shard must be requested."
}
foreach ($requested in $requestedShards) {
  if ($requested -notin $knownShards) {
    throw "Shard '$requested' is not present in inventory '$InventoryPath'."
  }
}

$selectedShards = @(
  foreach ($requested in $requestedShards) {
    $inventory.shards | Where-Object id -eq $requested | Select-Object -First 1
  }
)

$requestedBatches = @($Batch | Sort-Object -Unique)
foreach ($requestedBatch in $requestedBatches) {
  if ($requestedBatch -lt 1) {
    throw "Batch values are 1-based and must be greater than zero."
  }
}

$classNameRegex = $null
if (-not [string]::IsNullOrWhiteSpace($ClassNamePattern)) {
  try {
    $classNameRegex = [regex]::new($ClassNamePattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)
  }
  catch {
    throw "Invalid ClassNamePattern '$ClassNamePattern': $($_.Exception.Message)"
  }
}

$executionUnits = [Collections.Generic.List[object]]::new()
foreach ($shardDefinition in $selectedShards) {
  $shardId = [string]$shardDefinition.id
  $classes = @($shardDefinition.classes | ForEach-Object { [string]$_ })
  if ($null -ne $classNameRegex) {
    $classes = @($classes | Where-Object { $classNameRegex.IsMatch($_) })
  }
  if ($MissingOnly) {
    $classes = @($classes | Where-Object { $missingClassSet.Contains($_) })
  }
  if ($classes.Count -eq 0) {
    if ($MissingOnly) {
      Write-Host "Shard $shardId skipped: no missing classes selected."
      continue
    }
    throw "Shard '$shardId' has no classes after applying ClassNamePattern '$ClassNamePattern'."
  }

  $effectiveBatchSize = if ($BatchSize -gt 0) { $BatchSize } elseif ($MissingOnly) { 1 } else { $classes.Count }
  $batchCount = [int][Math]::Ceiling($classes.Count / [double]$effectiveBatchSize)
  for ($batchIndex = 0; $batchIndex -lt $batchCount; $batchIndex++) {
    $batchNumber = $batchIndex + 1
    if ($requestedBatches.Count -gt 0 -and $batchNumber -notin $requestedBatches) {
      continue
    }

    $batchClasses = @($classes | Select-Object -Skip ($batchIndex * $effectiveBatchSize) -First $effectiveBatchSize)
    $unitId = if ($BatchSize -gt 0 -or $MissingOnly) {
      "$shardId-batch-$($batchNumber.ToString('D2'))-of-$($batchCount.ToString('D2'))"
    }
    else {
      $shardId
    }
    $filterExpression = ($batchClasses | ForEach-Object { "FullyQualifiedName~$_" }) -join "|"
    $executionUnits.Add([pscustomobject][ordered]@{
        id = $unitId
        parentShardId = $shardId
        batchNumber = $batchNumber
        batchCount = $batchCount
        classCount = $batchClasses.Count
        classNames = @($batchClasses)
        filterExpression = $filterExpression
      })
  }
}

if ($executionUnits.Count -eq 0 -and -not $MissingOnly) {
  throw "No execution units were selected. Check -Shard, -BatchSize, -Batch, and -ClassNamePattern."
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$runDirectory = Join-Path $OutputRoot $RunId
New-Item -ItemType Directory -Force -Path $runDirectory | Out-Null

$results = [Collections.Generic.List[object]]::new()
foreach ($executionUnit in $executionUnits) {
  $shardId = [string]$executionUnit.id
  $parentShardId = [string]$executionUnit.parentShardId
  $batchNumber = [int]$executionUnit.batchNumber
  $batchCount = [int]$executionUnit.batchCount
  $classNames = @($executionUnit.classNames)
  $safeShardId = ($shardId.ToLowerInvariant() -replace '[^a-z0-9]+', '_').Trim('_')
  $classCount = [int]$executionUnit.classCount
  $filterExpression = [string]$executionUnit.filterExpression
  if ([string]::IsNullOrWhiteSpace($filterExpression)) {
    throw "Shard '$shardId' has no filter expression."
  }

  $logPath = Join-Path $runDirectory "project-quality-$safeShardId.log"
  $trxFileName = "project-quality-$safeShardId.trx"
  $trxPath = Join-Path $runDirectory $trxFileName
  $startedAt = [DateTime]::UtcNow

  $arguments = @(
    "test",
    $TestProject,
    "-c",
    $Configuration,
    "--no-build",
    "--filter",
    $filterExpression,
    "--logger",
    "console;verbosity=normal",
    "--logger",
    "trx;LogFileName=$trxFileName",
    "--results-directory",
    $runDirectory,
    "/p:UseSharedCompilation=false",
    "/nr:false"
  )
  $commandText = "dotnet " + ($arguments | ForEach-Object {
      if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    } | Join-String -Separator " ")

  if ($PreviewOnly) {
    $results.Add([pscustomobject][ordered]@{
      id = $shardId
        parentShardId = $parentShardId
        batchNumber = $batchNumber
        batchCount = $batchCount
        state = "preview"
        classCount = $classCount
        classNames = @($classNames)
        firstClass = [string]$classNames[0]
        lastClass = [string]$classNames[-1]
        filterExpressionLength = $filterExpression.Length
        command = $commandText
        timeoutSeconds = $TimeoutSeconds
        startedAtUtc = $startedAt.ToString("O")
        endedAtUtc = $startedAt.ToString("O")
        durationSeconds = 0
        exitCode = $null
        timedOut = $false
        processTreeCleanup = [pscustomobject][ordered]@{
          attempted = $false
          processExited = $true
          error = ""
        }
        counters = [pscustomobject][ordered]@{
          available = $false
          total = 0
          executed = 0
          passed = 0
          failed = 0
          error = 0
          timeout = 0
          aborted = 0
          inconclusive = 0
          notExecuted = 0
        }
        logPath = ""
        logSha256 = ""
        trxPath = ""
        trxSha256 = ""
        lastOutputLines = @()
      })
    Write-Host "Shard $shardId preview: parent=$parentShardId classes=$classCount filterLength=$($filterExpression.Length)"
    continue
  }

  $processInfo = [Diagnostics.ProcessStartInfo]::new()
  $processInfo.FileName = "dotnet"
  $processInfo.WorkingDirectory = $repositoryRoot
  $processInfo.UseShellExecute = $false
  $processInfo.CreateNoWindow = $true
  $processInfo.RedirectStandardOutput = $true
  $processInfo.RedirectStandardError = $true
  foreach ($argument in $arguments) {
    [void]$processInfo.ArgumentList.Add($argument)
  }
  $processInfo.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1"
  $processInfo.Environment["DOTNET_CLI_USE_MSBUILD_SERVER"] = "0"
  $processInfo.Environment["MSBUILDDISABLENODEREUSE"] = "1"

  $process = [Diagnostics.Process]::new()
  $process.StartInfo = $processInfo
  Write-Host "Shard $shardId started: classes=$classCount timeout=${TimeoutSeconds}s"
  if (-not $process.Start()) {
    throw "Failed to start dotnet test for shard '$shardId'."
  }

  $stdoutTask = $process.StandardOutput.ReadToEndAsync()
  $stderrTask = $process.StandardError.ReadToEndAsync()
  $waitTask = $process.WaitForExitAsync()
  $timedOut = -not $waitTask.Wait([TimeSpan]::FromSeconds($TimeoutSeconds))
  $cleanup = [pscustomobject][ordered]@{
    attempted = $false
    processExited = $process.HasExited
    error = ""
  }
  if ($timedOut) {
    $cleanup = Stop-ProcessTree -Process $process
  }
  elseif (-not $process.HasExited) {
    [void]$process.WaitForExit(30000)
  }

  $stdout = $stdoutTask.GetAwaiter().GetResult()
  $stderr = $stderrTask.GetAwaiter().GetResult()
  $endedAt = [DateTime]::UtcNow
  $durationSeconds = [Math]::Round(($endedAt - $startedAt).TotalSeconds, 3)
  $exitCode = if ($timedOut -or -not $process.HasExited) { -1 } else { $process.ExitCode }

  $transcript = [Collections.Generic.List[string]]::new()
  $transcript.Add("RecordKind=project-quality-test-shard-log")
  $transcript.Add("Shard=$shardId")
  $transcript.Add("StartedAtUtc=$($startedAt.ToString('O'))")
  $transcript.Add("EndedAtUtc=$($endedAt.ToString('O'))")
  $transcript.Add("DurationSeconds=$durationSeconds")
  $transcript.Add("TimeoutSeconds=$TimeoutSeconds")
  $transcript.Add("TimedOut=$timedOut")
  $transcript.Add("ExitCode=$exitCode")
  $transcript.Add("Command=$commandText")
  $transcript.Add("")
  $transcript.Add("===== STDOUT =====")
  $transcript.Add($stdout)
  $transcript.Add("")
  $transcript.Add("===== STDERR =====")
  $transcript.Add($stderr)
  $transcript | Set-Content -LiteralPath $logPath -Encoding utf8

  $counters = Get-TrxCounters -Path $trxPath
  $combinedOutput = @(
    ($stdout -split "\r?\n")
    ($stderr -split "\r?\n")
  ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  $lastOutputLines = @($combinedOutput | Select-Object -Last 25)
  $state = if ($timedOut) {
    "timed-out"
  }
  elseif ($exitCode -eq 0) {
    "passed"
  }
  else {
    "failed"
  }

  $results.Add([pscustomobject][ordered]@{
      id = $shardId
      parentShardId = $parentShardId
      batchNumber = $batchNumber
      batchCount = $batchCount
      state = $state
      classCount = $classCount
      classNames = @($classNames)
      firstClass = [string]$classNames[0]
      lastClass = [string]$classNames[-1]
      filterExpressionLength = $filterExpression.Length
      command = $commandText
      timeoutSeconds = $TimeoutSeconds
      startedAtUtc = $startedAt.ToString("O")
      endedAtUtc = $endedAt.ToString("O")
      durationSeconds = $durationSeconds
      exitCode = $exitCode
      timedOut = $timedOut
      processTreeCleanup = $cleanup
      counters = $counters
      logPath = ConvertTo-RelativePath $logPath
      logSha256 = (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()
      trxPath = if (Test-Path -LiteralPath $trxPath -PathType Leaf) { ConvertTo-RelativePath $trxPath } else { "" }
      trxSha256 = if (Test-Path -LiteralPath $trxPath -PathType Leaf) { (Get-FileHash -LiteralPath $trxPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { "" }
      lastOutputLines = $lastOutputLines
    })

  Write-Host "Shard $shardId finished: state=$state exitCode=$exitCode duration=${durationSeconds}s total=$($counters.total) passed=$($counters.passed) failed=$($counters.failed)"
}

$passedResults = @($results | Where-Object state -eq "passed")
$failedResults = @($results | Where-Object state -eq "failed")
$timedOutResults = @($results | Where-Object state -eq "timed-out")
$previewResults = @($results | Where-Object state -eq "preview")
$selectedClassNames = @($executionUnits | ForEach-Object classNames | ForEach-Object { [string]$_ } | Sort-Object -Unique)
$classLevelExecutionUnitCount = @($results | Where-Object classCount -eq 1).Count
$slowestExecutionUnits = @(
  $results |
    Sort-Object @{ Expression = { [double]$_.durationSeconds }; Descending = $true }, id |
    Select-Object -First $DurationRankingCount |
    ForEach-Object {
      [pscustomobject][ordered]@{
        id = [string]$_.id
        state = [string]$_.state
        classCount = [int]$_.classCount
        classNames = @($_.classNames)
        durationSeconds = [double]$_.durationSeconds
        timedOut = [bool]$_.timedOut
        processTreeCleanup = $_.processTreeCleanup
        trxPath = [string]$_.trxPath
        trxSha256 = [string]$_.trxSha256
      }
    }
)
$summaryState = if ($PreviewOnly) {
  "preview"
}
elseif ($failedResults.Count -eq 0 -and $timedOutResults.Count -eq 0) {
  "passed"
}
else {
  "failed"
}

$summary = [pscustomobject][ordered]@{
  recordKind = "project-quality-test-shard-run-summary"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  runId = $RunId
  runState = $summaryState
  previewOnly = [bool]$PreviewOnly
  configuration = $Configuration
  testProject = ConvertTo-RelativePath $TestProject
  inventoryPath = ConvertTo-RelativePath $InventoryPath
  inventorySha256 = $inventorySha256
  missingOnly = [bool]$MissingOnly
  coveragePath = if ($MissingOnly) { ConvertTo-RelativePath $CoveragePath } else { "" }
  coverageSha256 = $coverageSha256
  requestedMissingClassCount = $requestedMissingClasses.Count
  selectedMissingClassCount = if ($MissingOnly) { $selectedClassNames.Count } else { 0 }
  selectedClassNames = @($selectedClassNames)
  timeoutSecondsPerShard = $TimeoutSeconds
  requestedShards = @($requestedShards)
  requestedShardCount = $requestedShards.Count
  batchSize = $BatchSize
  requestedBatches = @($requestedBatches)
  classNamePattern = $ClassNamePattern
  durationRankingCount = $DurationRankingCount
  executionUnitCount = $results.Count
  classLevelExecutionUnitCount = $classLevelExecutionUnitCount
  shardCount = $results.Count
  passedShardCount = $passedResults.Count
  failedShardCount = $failedResults.Count
  timedOutShardCount = $timedOutResults.Count
  previewShardCount = $previewResults.Count
  totalExecutedTests = [int](($results | ForEach-Object { $_.counters.executed } | Measure-Object -Sum).Sum)
  totalPassedTests = [int](($results | ForEach-Object { $_.counters.passed } | Measure-Object -Sum).Sum)
  totalFailedTests = [int](($results | ForEach-Object { $_.counters.failed } | Measure-Object -Sum).Sum)
  slowestExecutionUnits = @($slowestExecutionUnits)
  results = @($results)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  boundary = "Shard execution is source quality evidence only. Preview and inventory are not test passes. A timed-out shard remains failed until its blocking class or process lifecycle is fixed."
}

$summaryJsonPath = Join-Path $runDirectory "summary.json"
$summaryMarkdownPath = Join-Path $runDirectory "summary.md"
$latestJsonPath = Join-Path $OutputRoot "latest-summary.json"
$latestMarkdownPath = Join-Path $OutputRoot "latest-summary.md"
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryJsonPath -Encoding utf8
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $latestJsonPath -Encoding utf8

$markdown = [Collections.Generic.List[string]]::new()
$markdown.Add("# ProjectQuality 测试分片执行摘要")
$markdown.Add("")
$markdown.Add("- Run ID：``$RunId``")
$markdown.Add("- 状态：``$summaryState``")
$markdown.Add("- Preview：``$([bool]$PreviewOnly)``")
$markdown.Add("- 每分片超时：``$TimeoutSeconds`` 秒")
$markdown.Add("- 分片：``$($requestedShards -join ', ')``")
$markdown.Add("- 批次大小：``$BatchSize``（0 表示整个分片）")
$markdown.Add("- 指定批次：``$(if ($requestedBatches.Count -gt 0) { $requestedBatches -join ', ' } else { '全部' })``")
$markdown.Add("- 类名筛选：``$(if ([string]::IsNullOrWhiteSpace($ClassNamePattern)) { '无' } else { $ClassNamePattern })``")
$markdown.Add("- 仅续跑缺失类：``$([bool]$MissingOnly)``")
$markdown.Add("- coverage：``$(if ($MissingOnly) { ConvertTo-RelativePath $CoveragePath } else { '未使用' })``")
$markdown.Add("- 请求缺失类：``$($requestedMissingClasses.Count)``")
$markdown.Add("- 选中缺失类：``$(if ($MissingOnly) { $selectedClassNames.Count } else { 0 })``")
$markdown.Add("- 执行单元：``$($summary.executionUnitCount)``")
$markdown.Add("- 单类执行单元：``$($summary.classLevelExecutionUnitCount)``")
$markdown.Add("- 通过分片：``$($summary.passedShardCount)``")
$markdown.Add("- 失败分片：``$($summary.failedShardCount)``")
$markdown.Add("- 超时分片：``$($summary.timedOutShardCount)``")
$markdown.Add("- 已执行测试：``$($summary.totalExecutedTests)``")
$markdown.Add("- 通过测试：``$($summary.totalPassedTests)``")
$markdown.Add("- 失败测试：``$($summary.totalFailedTests)``")
$markdown.Add("")
$markdown.Add("| 分片 | 状态 | 类 | 时长（秒） | Exit | Total | Passed | Failed | 日志 |")
$markdown.Add("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
foreach ($result in $results) {
  $markdown.Add("| ``$($result.id)`` | ``$($result.state)`` | $($result.classCount) | $($result.durationSeconds) | $($result.exitCode) | $($result.counters.total) | $($result.counters.passed) | $($result.counters.failed) | ``$($result.logPath)`` |")
}
$markdown.Add("")
$markdown.Add("## 耗时排名")
$markdown.Add("")
$markdown.Add("| 执行单元 | 状态 | 类 | 时长（秒） | 超时 | kill-tree |")
$markdown.Add("| --- | --- | ---: | ---: | --- | --- |")
foreach ($result in $slowestExecutionUnits) {
  $markdown.Add("| ``$($result.id)`` | ``$($result.state)`` | $($result.classCount) | $($result.durationSeconds) | $($result.timedOut) | $($result.processTreeCleanup.attempted) |")
}
$markdown.Add("")
$markdown.Add("## 边界")
$markdown.Add("")
$markdown.Add($summary.boundary)
$markdown | Set-Content -LiteralPath $summaryMarkdownPath -Encoding utf8
$markdown | Set-Content -LiteralPath $latestMarkdownPath -Encoding utf8

Write-Host "ProjectQuality shard summary written."
Write-Host "JSON=$summaryJsonPath"
Write-Host "Markdown=$summaryMarkdownPath"
Write-Host "RunState=$summaryState Shards=$($results.Count) Passed=$($summary.passedShardCount) Failed=$($summary.failedShardCount) TimedOut=$($summary.timedOutShardCount)"
Write-Host "TestsExecuted=$($summary.totalExecutedTests) Passed=$($summary.totalPassedTests) Failed=$($summary.totalFailedTests)"

if (-not $PreviewOnly -and -not $ContinueOnFailure -and $summaryState -ne "passed") {
  exit 1
}
