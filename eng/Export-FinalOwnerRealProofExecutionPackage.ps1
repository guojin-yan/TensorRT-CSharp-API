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

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  $content = (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine)
  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $content, $script:utf8)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
}

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-SourceSummary {
  param([string]$Id, [string]$Artifact, [AllowNull()][object]$Record, [string]$StateProperty, [string]$DefaultState)
  [pscustomobject]@{
    id = $Id
    artifact = $Artifact
    present = $null -ne $Record
    state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
    ownerActionRequired = [bool](Get-PropertyOrDefault -Object $Record -Name "ownerActionRequired" -DefaultValue $true)
    passed = [bool](Get-PropertyOrDefault -Object $Record -Name "passed" -DefaultValue $false)
    proofCandidateReady = [bool](Get-PropertyOrDefault -Object $Record -Name "proofCandidateReady" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
  }
}

function New-ExecutionStep {
  param([int]$Order, [string]$Id, [string]$Title, [string]$OwnerCommand, [string]$RequiredEvidence, [string]$Boundary)
  [pscustomobject]@{
    order = $Order
    id = $Id
    title = $Title
    ownerCommand = $OwnerCommand
    requiredEvidence = $RequiredEvidence
    ownerActionRequired = $true
    performsPublishInAutomation = $false
    performsRuntimeExecutionInAutomation = $false
    canPromoteProofByItself = $false
    boundary = $Boundary
  }
}

$workspaceContract = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-workspace-contract.json"
$workspaceContractValidation = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-workspace-contract-validation.json"
$ownerCommandPack = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-owner-command-pack.json"
$ownerCommandPackValidation = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-owner-command-pack-validation.json"
$externalImport = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-import.json"
$externalCandidate = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-candidate.json"
$externalValidation = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-validation.json"
$postPublishImport = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-import.json"
$postPublishCandidate = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-candidate.json"
$postPublishValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"
$closeReadiness = Read-JsonOrNull "artifacts\final-release\final-owner-execution-close-readiness-from-real-input.json"
$closeReadinessValidation = Read-JsonOrNull "artifacts\final-release\final-owner-execution-close-readiness-from-real-input-validation.json"

$sourceSummaries = @(
  New-SourceSummary "external-clean-consumer-workspace-contract" "artifacts/final-release/external-clean-consumer-execution-workspace-contract.json" $workspaceContract "contractState" "missing-external-clean-consumer-execution-workspace-contract"
  New-SourceSummary "external-clean-consumer-workspace-contract-validation" "artifacts/final-release/external-clean-consumer-execution-workspace-contract-validation.json" $workspaceContractValidation "validationState" "missing-external-clean-consumer-execution-workspace-contract-validation"
  New-SourceSummary "external-clean-consumer-owner-command-pack" "artifacts/final-release/external-clean-consumer-owner-command-pack.json" $ownerCommandPack "packState" "missing-external-clean-consumer-owner-command-pack"
  New-SourceSummary "external-clean-consumer-owner-command-pack-validation" "artifacts/final-release/external-clean-consumer-owner-command-pack-validation.json" $ownerCommandPackValidation "validationState" "missing-external-clean-consumer-owner-command-pack-validation"
  New-SourceSummary "external-clean-consumer-execution-result-import" "artifacts/final-release/external-clean-consumer-execution-result-import.json" $externalImport "importState" "missing-external-clean-consumer-execution-result-import"
  New-SourceSummary "external-clean-consumer-execution-result-candidate" "artifacts/final-release/external-clean-consumer-execution-result-candidate.json" $externalCandidate "candidateState" "missing-external-clean-consumer-execution-result-candidate"
  New-SourceSummary "external-clean-consumer-execution-result-validation" "artifacts/final-release/external-clean-consumer-execution-result-validation.json" $externalValidation "validationState" "missing-external-clean-consumer-execution-result-validation"
  New-SourceSummary "post-publish-clean-consumer-proof-result-import" "artifacts/final-release/post-publish-clean-consumer-proof-result-import.json" $postPublishImport "importState" "missing-post-publish-clean-consumer-proof-result-import"
  New-SourceSummary "post-publish-clean-consumer-proof-result-candidate" "artifacts/final-release/post-publish-clean-consumer-proof-result-candidate.json" $postPublishCandidate "candidateState" "missing-post-publish-clean-consumer-proof-result-candidate"
  New-SourceSummary "post-publish-clean-consumer-proof-result-validation" "artifacts/final-release/post-publish-clean-consumer-proof-result-validation.json" $postPublishValidation "validationState" "missing-post-publish-clean-consumer-proof-result-validation"
  New-SourceSummary "final-owner-execution-close-readiness-from-real-input" "artifacts/final-release/final-owner-execution-close-readiness-from-real-input.json" $closeReadiness "readinessState" "missing-final-owner-execution-close-readiness-from-real-input"
  New-SourceSummary "final-owner-execution-close-readiness-from-real-input-validation" "artifacts/final-release/final-owner-execution-close-readiness-from-real-input-validation.json" $closeReadinessValidation "validationState" "missing-final-owner-execution-close-readiness-from-real-input-validation"
)

$executionSteps = @(
  New-ExecutionStep 1 "create-repository-external-clean-consumer-workspace" "Create repository-external CleanConsumer workspace" 'New-Item -ItemType Directory -Force -Path "<owner-external-root>\TensorRtSharpCleanConsumer"; Set-Location "<owner-external-root>\TensorRtSharpCleanConsumer"' "Absolute workspace path outside the TensorRtSharp repository." "Workspace contract only; not runtime proof."
  New-ExecutionStep 2 "restore-from-real-public-package-source" "Restore from real/public package source" 'dotnet new console -n TensorRtSharpCleanConsumer --framework net8.0; dotnet add .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj package JYPPX.TensorRtSharp --version <owner-version> --source <owner-real-package-source-url>; dotnet restore .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj *> .\restore.log' "restore.log, package source URL, package identities, and package hashes." "Local feed, ProjectReference, and direct .nupkg are forbidden substitutes."
  New-ExecutionStep 3 "build-clean-consumer" "Build CleanConsumer" 'dotnet build .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj -c Release --no-restore *> .\build.log' "build.log and buildLogSha256." "Build-only output cannot promote proof."
  New-ExecutionStep 4 "run-clean-consumer-smoke" "Run CleanConsumer smoke" 'dotnet run --project .\TensorRtSharpCleanConsumer\TensorRtSharpCleanConsumer.csproj -c Release --no-build 1> .\smoke.stdout.log 2> .\smoke.stderr.log; $LASTEXITCODE | Set-Content .\smoke.exitcode.txt' "exitCode=0, stdout/stderr logs, run log, and hashes." "This package only describes owner execution; it does not execute smoke in automation."
  New-ExecutionStep 5 "collect-logs-and-native-assets" "Collect stdout/stderr/log/native asset listing" 'Get-ChildItem -Recurse -File .\TensorRtSharpCleanConsumer\bin\Release | Where-Object { $_.Name -match "jyppx|tensorrt|cuda|cudnn|\.dll$|\.so$" } | Select-Object FullName,Length | ConvertTo-Json -Depth 5 | Set-Content .\native-assets.json' "native-assets.json, smoke stdout/stderr, restore/build logs." "Native asset listing alone is not runtime proof."
  New-ExecutionStep 6 "compute-sha256-manifest" "Compute SHA256 for all evidence files" 'Get-ChildItem .\restore.log,.\build.log,.\smoke.stdout.log,.\smoke.stderr.log,.\native-assets.json -File | Get-FileHash -Algorithm SHA256 | ConvertTo-Json -Depth 5 | Set-Content .\sha256-manifest.json' "SHA256 manifest covering every evidence file and package." "Hashes must later be checked by strict import."
  New-ExecutionStep 7 "fill-external-clean-consumer-owner-input" "Fill external CleanConsumer owner input" 'Copy-Item "<repo>\artifacts\final-release\external-clean-consumer-execution-result.template.json" .\external-clean-consumer-execution-result.owner.json; <owner-fill-all-real-paths-hashes-host-metadata-and-confirmations>' "external-clean-consumer-execution-result.owner.json with no placeholders." "Template remains non-proof until real owner evidence is supplied."
  New-ExecutionStep 8 "run-external-clean-consumer-import-and-strict-validator" "Run External CleanConsumer import + strict validator" 'pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Import-ExternalCleanConsumerExecutionResult.ps1" -OwnerInputPath .\external-clean-consumer-execution-result.owner.json -RequireExistingFiles -RequireHashMatch; pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Test-ExternalCleanConsumerExecutionResult.ps1" -Strict -RequireExistingFiles -RequireHashMatch -FailOnNotProof' "accepted import, candidate, and validation artifacts." "failedBlockerCount=0 alone is insufficient; proofCandidateReady must be true."
  New-ExecutionStep 9 "download-public-published-packages" "After public publish, download real packages" 'dotnet nuget locals all --clear; <owner-download-from-public-source-only>; Get-FileHash -Algorithm SHA256 <downloaded-packages> | ConvertTo-Json -Depth 5 | Set-Content .\post-publish-package-hashes.json' "public package URLs, downloaded package files, and SHA256 values." "This package does not run dotnet nuget push."
  New-ExecutionStep 10 "fill-post-publish-owner-input" "Fill post-publish proof owner input" 'Copy-Item "<repo>\artifacts\final-release\post-publish-clean-consumer-proof-result.template.json" .\post-publish-clean-consumer-proof-result.owner.json; <owner-fill-public-package-source-download-hashes-clean-consumer-logs-and-confirmations>' "post-publish owner input JSON with public package evidence." "Pre-publish smoke cannot substitute post-publish proof."
  New-ExecutionStep 11 "run-post-publish-import-and-strict-validator" "Run PostPublish import + strict validator" 'pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Import-PostPublishCleanConsumerProofResult.ps1" -OwnerInputPath .\post-publish-clean-consumer-proof-result.owner.json -RequireExistingFiles -RequireHashMatch; pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Test-PostPublishCleanConsumerProofResult.ps1" -Strict -RequireExistingFiles -RequireHashMatch -FailOnNotProof' "accepted post-publish proof import/candidate/validation artifacts." "PostPublish candidate is not proof until strict validator accepts real public-source evidence."
  New-ExecutionStep 12 "fill-rollback-review" "Fill rollback review" '<owner-fill-rollback-review-artifact>' "rollback review artifact accepted by release close workflow." "Rollback review is release governance evidence, not runtime proof."
  New-ExecutionStep 13 "fill-final-close-decision" "Fill final close decision" '<owner-fill-final-close-decision-artifact>' "final close decision artifact accepted by release close workflow." "Final close decision is owner approval only after proof lanes pass."
  New-ExecutionStep 14 "refresh-release-evidence-classification-and-close-readiness" "Refresh release evidence / classification audit / close readiness" 'pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Test-FinalOwnerExecutionCloseReadinessFromRealInput.ps1" -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Export-ReleaseEvidenceBundle.ps1"; pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Test-ReleaseEvidenceClassificationAudit.ps1" -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File "<repo>\eng\Test-FinalOwnerRealProofConvergenceGate.ps1" -Strict' "refreshed release evidence, classification audit, close readiness, and convergence gate." "Refresh commands aggregate evidence only and cannot fabricate proof."
)

$requiredOwnerInputs = @(
  "repositoryExternalWorkspaceRoot",
  "realPackageSourceUrl",
  "cleanConsumerCsprojPath",
  "restoreLogPath",
  "buildLogPath",
  "runLogPath",
  "smokeStdoutPath",
  "smokeStderrPath",
  "nativeAssetListingPath",
  "hostMetadata",
  "managedPackageSha256",
  "runtimePackageSha256",
  "allEvidenceFileSha256",
  "externalCleanConsumerExecutionResultOwnerJson",
  "postPublishCleanConsumerProofResultOwnerJson",
  "publicPackageUrl",
  "publicDownloadedPackageSha256",
  "rollbackReviewArtifact",
  "finalCloseDecisionArtifact",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)

$forbiddenSubstitutes = @(
  "local smoke",
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "direct nupkg",
  "build-only",
  "dependency-probe",
  "dashboard",
  "runbook",
  "template",
  "candidate",
  "command pack",
  "contract",
  "gap matrix",
  "failedBlockerCount=0",
  "pre-publish smoke reused as post-publish proof"
)

$record = [pscustomobject]@{
  recordKind = "final-owner-real-proof-execution-package"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packageState = "blocked-final-owner-real-proof-execution-required"
  sourceSummaryCount = $sourceSummaries.Count
  executionStepCount = $executionSteps.Count
  requiredOwnerInputCount = $requiredOwnerInputs.Count
  forbiddenSubstituteCount = $forbiddenSubstitutes.Count
  sourceSummaries = @($sourceSummaries)
  ownerExecutionSequence = @($executionSteps)
  requiredOwnerInputs = @($requiredOwnerInputs)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner real proof execution package is owner execution guidance and evidence routing only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-real-proof-execution-package.json"
$markdownPath = Join-Path $OutputRoot "final-owner-real-proof-execution-package.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 16)
$summaryRows = foreach ($summary in $sourceSummaries) {
  "| ``$($summary.id)`` | ``$($summary.present)`` | ``$(ConvertTo-MarkdownCell $summary.state)`` | ``$($summary.proofCandidateReady)`` |"
}
$stepRows = foreach ($step in $executionSteps) {
  "| $($step.order) | ``$($step.id)`` | $(ConvertTo-MarkdownCell $step.title) | $(ConvertTo-MarkdownCell $step.requiredEvidence) |"
}

$markdown = @"
# Final Owner Real Proof Execution Package

| Field | Value |
|---|---|
| packageState | ``$($record.packageState)`` |
| executionStepCount | ``$($record.executionStepCount)`` |
| requiredOwnerInputCount | ``$($record.requiredOwnerInputCount)`` |
| ownerActionRequired | ``$($record.ownerActionRequired)`` |
| passed | ``$($record.passed)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Source Summary

| ID | Present | State | Proof Ready |
|---|---:|---|---:|
$($summaryRows -join "`r`n")

## Owner Execution Sequence

| Order | ID | Title | Required Evidence |
|---:|---|---|---|
$($stepRows -join "`r`n")

## Boundary

$($record.boundary)
"@

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"
