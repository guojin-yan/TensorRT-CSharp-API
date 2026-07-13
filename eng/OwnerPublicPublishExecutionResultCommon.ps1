$script:OwnerPublicPublishUtf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $script:OwnerPublicPublishUtf8
$OutputEncoding = $script:OwnerPublicPublishUtf8

function Initialize-OwnerPublicPublishContext {
  param(
    [string]$RepositoryRoot,
    [string]$OutputDirectory
  )

  if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
    $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
  }

  if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
  }
  elseif (-not [System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $RepositoryRoot $OutputDirectory
  }

  New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

  [pscustomobject]@{
    RepositoryRoot = $RepositoryRoot
    OutputDirectory = $OutputDirectory
  }
}

function Resolve-OwnerPath {
  param([string]$RepositoryRoot, [string]$Path)
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function ConvertTo-OwnerFlatStringLines {
  param([AllowNull()][object]$Value)

  foreach ($item in @($Value)) {
    if ($null -eq $item) {
      ""
    }
    elseif ($item -is [string]) {
      $item
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { [string]$child }
    }
    else {
      [string]$item
    }
  }
}

function Write-OwnerUtf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)

  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $lines = @(ConvertTo-OwnerFlatStringLines -Value $InputObject)
  [IO.File]::WriteAllText($LiteralPath, (($lines -join [Environment]::NewLine) + [Environment]::NewLine), $script:OwnerPublicPublishUtf8)
}

function Read-OwnerJsonOrNull {
  param([string]$RepositoryRoot, [string]$Path)

  $resolvedPath = Resolve-OwnerPath -RepositoryRoot $RepositoryRoot -Path $Path
  if (-not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-OwnerPropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-OwnerMarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function New-OwnerPublicPublishRequiredField {
  param([string]$Group, [string]$Name, [string[]]$ForbiddenSubstitutes = @())

  [pscustomobject]@{
    group = $Group
    name = $Name
    description = "Owner supplied real public publish evidence field: $Name."
    required = $true
    valueState = "owner-input-required"
    ready = $false
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
  }
}

function Get-OwnerPublicPublishRequiredFields {
  $commonSubstitutes = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "build-only",
    "dependency-probe-only",
    "dry-run-only",
    "candidate",
    "dashboard",
    "runbook",
    "manual approval",
    "queued GitHub Actions run",
    "missing self-hosted runner",
    "sidecar-only",
    "TensorRtExec report"
  )
  $groups = [ordered]@{
    publicPackage = @(
      "publicPackageId", "publicPackageVersion", "publicPackageSource", "publicPackageUrl", "publicPackageDownloadedPath",
      "publicPackageSha256", "publicPackagePublishedAtUtc", "publicPackageOwnerAccount", "publicPackageVisibility",
      "publicPackageLicense", "publicPackageTags", "publicPackageDescriptionHash", "managedPackageId", "runtimePackageId",
      "managedPackageUrl", "runtimePackageUrl", "managedPackageSha256", "runtimePackageSha256"
    )
    githubReleaseAsset = @(
      "githubReleaseUrl", "githubReleaseTag", "githubReleaseAssetName", "githubReleaseAssetUrl", "githubReleaseAssetDownloadedPath",
      "githubReleaseAssetSha256", "githubReleaseUploadTranscriptPath", "githubReleaseUploadTranscriptSha256",
      "githubReleaseWasNotUploadedExplanation", "githubReleaseReviewer"
    )
    nugetPush = @(
      "nugetPushCommand", "nugetPushSource", "nugetPushTranscriptPath", "nugetPushTranscriptSha256", "nugetPushStdoutPath",
      "nugetPushStdoutSha256", "nugetPushStderrPath", "nugetPushStderrSha256", "nugetPushMergedTranscriptPath",
      "nugetPushMergedTranscriptSha256", "nugetPushExitCode", "nugetPushWasNotPerformedExplanation", "publishCommandPlanPath",
      "publishCommandPlanSha256", "managedPublishCommandSha256", "runtimePublishCommandSha256"
    )
    cleanConsumerRestore = @(
      "cleanConsumerRestoreRoot", "cleanConsumerRestoreCommand", "cleanConsumerRestoreStdoutPath", "cleanConsumerRestoreStdoutSha256",
      "cleanConsumerRestoreStderrPath", "cleanConsumerRestoreStderrSha256", "cleanConsumerRestoreMergedTranscriptPath",
      "cleanConsumerRestoreMergedTranscriptSha256", "cleanConsumerRestoreExitCode", "cleanConsumerRestorePackageSource",
      "cleanConsumerRestoreNoLocalFeedEvidence", "cleanConsumerRestoreLockFileHash"
    )
    cleanConsumerBuild = @(
      "cleanConsumerBuildRoot", "cleanConsumerBuildProjectPath", "cleanConsumerBuildCommand", "cleanConsumerBuildStdoutPath",
      "cleanConsumerBuildStdoutSha256", "cleanConsumerBuildStderrPath", "cleanConsumerBuildStderrSha256",
      "cleanConsumerBuildMergedTranscriptPath", "cleanConsumerBuildMergedTranscriptSha256", "cleanConsumerBuildExitCode",
      "cleanConsumerBuildBinlogPath", "cleanConsumerBuildBinlogSha256"
    )
    cleanConsumerRuntimeSmoke = @(
      "cleanConsumerRuntimeSmokeRoot", "cleanConsumerRuntimeSmokeCommand", "cleanConsumerRuntimeSmokeStdoutPath",
      "cleanConsumerRuntimeSmokeStdoutSha256", "cleanConsumerRuntimeSmokeStderrPath", "cleanConsumerRuntimeSmokeStderrSha256",
      "cleanConsumerRuntimeSmokeMergedTranscriptPath", "cleanConsumerRuntimeSmokeMergedTranscriptSha256",
      "cleanConsumerRuntimeSmokeExitCode", "cleanConsumerRuntimeSmokeReportPath", "cleanConsumerRuntimeSmokeReportSha256",
      "cleanConsumerRuntimeSmokeTensorRtEngineHash"
    )
    strictValidator = @(
      "strictValidatorCommand", "strictValidatorOutputPath", "strictValidatorOutputSha256", "strictValidatorStdoutPath",
      "strictValidatorStdoutSha256", "strictValidatorStderrPath", "strictValidatorStderrSha256", "strictValidatorExitCode",
      "strictValidatorMergedTranscriptPath", "strictValidatorMergedTranscriptSha256"
    )
    hostIdentity = @(
      "hostMachineName", "hostOs", "hostOsVersion", "hostArchitecture", "hostCpuName", "hostCpuCoreCount", "hostGpuName",
      "hostGpuMemory", "hostGpuDriverVersion", "hostCudaVersion", "hostTensorRtVersion", "hostCudnnVersion", "hostDotnetSdkVersion",
      "hostDotnetRuntimeVersion", "hostPowerShellVersion", "hostEnvironmentTranscriptPath", "hostEnvironmentTranscriptSha256",
      "sourceRunnerQueueStatus", "sourceRunnerInfrastructureStatus", "sourceRunnerOwnerAction"
    )
    packageIdentity = @(
      "packageManagedPackageId", "packageManagedPackageVersion", "packageManagedPackageSourceChannel", "packageManagedPackageUrl",
      "packageManagedPackageSha256", "packageNativeBridgePackageId", "packageNativeBridgePackageVersion",
      "packageNativeBridgePackageUrl", "packageNativeBridgePackageSha256", "packageRuntimePackageId",
      "packageRuntimePackageVersion", "packageRuntimePackageUrl", "packageRuntimePackageSha256", "packageSourceChannel",
      "packageDependencyGraphPath", "packageDependencyGraphSha256"
    )
    releaseNotes = @(
      "releaseNotesPath", "releaseNotesSha256", "releaseNotesPublicUrl", "releaseNotesReviewer", "releaseNotesReviewedAtUtc",
      "releaseNotesApprovalId", "releaseNotesChangelogPath", "releaseNotesChangelogSha256"
    )
    ownerDecision = @(
      "rollbackDecision", "rollbackDecisionReason", "rollbackPlanPath", "rollbackPlanSha256", "noRollbackApprovalId",
      "releaseIssueCloseDecision", "releaseIssueCloseDecisionReason", "releaseIssueUrl", "releaseIssueCloseApprovalId",
      "finalPublicPackageUrlApproval", "finalPublicPackageHashApproval", "finalPublicPackageApprovalTimestampUtc"
    )
    ownerApproval = @(
      "ownerReviewer", "ownerReviewerEmail", "ownerReviewTimestampUtc", "ownerSignature", "ownerApprovalId",
      "ownerApprovalTranscriptPath", "ownerApprovalTranscriptSha256", "ownerApprovalScope", "ownerApprovalNotesPath",
      "ownerApprovalNotesSha256", "ownerAuthorizationId", "ownerAuthorizationScope", "ownerAuthorizationTimestampUtc"
    )
    nonSubstituteConfirmations = @(
      "noLocalFeedConfirmation", "noProjectReferenceConfirmation", "noDirectNupkgConfirmation", "noBuildOnlyConfirmation",
      "noDependencyProbeOnlyConfirmation", "noDryRunOnlyConfirmation", "noCandidateDashboardRunbookSubstitutionConfirmation",
      "noTemplateSubstitutionConfirmation", "noDraftSubstitutionConfirmation", "noBlockedDashboardSubstitutionConfirmation",
      "noLocalArtifactScanSubstitutionConfirmation", "noFakePackageUrlOrHashConfirmation", "noManualApprovalOnlyConfirmation",
      "noQueuedWorkflowConfirmation", "noMissingSelfHostedRunnerConfirmation", "noSidecarOnlyConfirmation",
      "noTensorRtExecReportOnlyConfirmation", "forbiddenSubstituteScanPath", "forbiddenSubstituteScanSha256"
    )
  }

  $fields = New-Object System.Collections.Generic.List[object]
  foreach ($group in $groups.Keys) {
    $forbidden = if ($group -in @("cleanConsumerRestore", "cleanConsumerBuild", "cleanConsumerRuntimeSmoke", "nonSubstituteConfirmations")) { $commonSubstitutes } else { @() }
    foreach ($name in $groups[$group]) {
      $fields.Add((New-OwnerPublicPublishRequiredField -Group $group -Name $name -ForbiddenSubstitutes $forbidden)) | Out-Null
    }
  }

  return @($fields.ToArray())
}

function Test-OwnerInputValueReady {
  param([string]$Name, [AllowNull()][object]$Value)

  if ($null -eq $Value) { return $false }
  $text = ([string]$Value).Trim()
  if ([string]::IsNullOrWhiteSpace($text)) { return $false }
  if ($text.StartsWith("<owner-fill", [StringComparison]::OrdinalIgnoreCase)) { return $false }
  if ($text.IndexOf("<owner-", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("template", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("local feed", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("ProjectReference", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("direct .nupkg", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("dry-run", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("dashboard", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("manual approval", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("queued GitHub Actions run", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("missing self-hosted runner", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("sidecar-only", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($text.IndexOf("TensorRtExec report", [StringComparison]::OrdinalIgnoreCase) -ge 0) { return $false }
  if ($Name -match "Sha256$" -and $text -notmatch "^[a-fA-F0-9]{64}$") { return $false }
  if ($Name -match "Url$" -and $text -notmatch "^https?://") { return $false }
  return $true
}

function ConvertTo-OwnerPublicPublishFieldMap {
  param([AllowNull()][object[]]$Fields)

  $map = [ordered]@{}
  foreach ($field in @($Fields)) {
    $name = [string](Get-OwnerPropertyOrDefault -Object $field -Name "name" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($name)) { continue }
    $map[$name] = [string](Get-OwnerPropertyOrDefault -Object $field -Name "value" -DefaultValue "")
  }

  return $map
}

function Get-OwnerPublicPublishFieldValue {
  param(
    [AllowNull()][object]$FieldMap,
    [string]$Name
  )

  if ($null -eq $FieldMap) { return "" }
  if ($FieldMap -is [System.Collections.IDictionary] -and $FieldMap.Contains($Name)) { return [string]$FieldMap[$Name] }
  if ($FieldMap.PSObject.Properties.Name -contains $Name) { return [string]$FieldMap.PSObject.Properties[$Name].Value }
  return ""
}

function Test-OwnerSha256Format {
  param([string]$Value)
  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -match '^[0-9a-fA-F]{64}$'
}

function Test-OwnerHttpUrl {
  param([string]$Value)
  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -match '^https?://'
}

function Test-OwnerDateTimeOffset {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
  $parsed = [DateTimeOffset]::MinValue
  return [DateTimeOffset]::TryParse(
    $Value,
    [System.Globalization.CultureInfo]::InvariantCulture,
    [System.Globalization.DateTimeStyles]::AssumeUniversal,
    [ref]$parsed)
}

function Test-OwnerPublicNuGetUrl {
  param([string]$Value)
  return (Test-OwnerHttpUrl -Value $Value) -and $Value.StartsWith("https://www.nuget.org/packages/", [StringComparison]::OrdinalIgnoreCase)
}

function Test-OwnerGitHubUrl {
  param([string]$Value)
  return (Test-OwnerHttpUrl -Value $Value) -and $Value.StartsWith("https://github.com/", [StringComparison]::OrdinalIgnoreCase)
}

function Get-OwnerPublicPublishForbiddenFindings {
  param(
    [AllowNull()][object]$Record,
    [AllowNull()][object]$FieldMap
  )

  $findings = New-Object System.Collections.Generic.List[string]
  if ($null -eq $Record) { return @() }

  $pieces = New-Object System.Collections.Generic.List[string]
  foreach ($property in $Record.PSObject.Properties) {
    if ($property.Value -is [string]) { $pieces.Add([string]$property.Value) | Out-Null }
  }
  if ($null -ne $FieldMap) {
    if ($FieldMap -is [System.Collections.IDictionary]) {
      foreach ($key in $FieldMap.Keys) { $pieces.Add([string]$FieldMap[$key]) | Out-Null }
    }
    else {
      foreach ($property in $FieldMap.PSObject.Properties) { $pieces.Add([string]$property.Value) | Out-Null }
    }
  }

  $text = ($pieces.ToArray() -join "`n")
  foreach ($pattern in @(
      @{ id = "local-feed"; regex = '(?i)local\s+feed|local-feed|file://|\\local-feed\\|/local-feed/' },
      @{ id = "project-reference"; regex = '(?i)projectreference|project\s+reference' },
      @{ id = "direct-nupkg"; regex = '(?i)direct\s+\.?nupkg|direct-nupkg' },
      @{ id = "package-managed-dry-run"; regex = '(?i)package-managed-dry-run' },
      @{ id = "manual-approval"; regex = '(?i)manual\s+approval' },
      @{ id = "queued-workflow"; regex = '(?i)queued\s+(github\s+actions\s+)?workflow|queued\s+github\s+actions\s+run' },
      @{ id = "missing-runner"; regex = '(?i)missing\s+(self-hosted\s+)?runner' },
      @{ id = "dashboard-only"; regex = '(?i)dashboard-only|dashboard\s+only' },
      @{ id = "artifact-only"; regex = '(?i)artifact-only|artifact\s+only' },
      @{ id = "local-dotnet-test"; regex = '(?i)local\s+dotnet\s+test' },
      @{ id = "sidecar-only"; regex = '(?i)sidecar-only|sidecar\s+only' },
      @{ id = "tensorrtexec-report"; regex = '(?i)tensorrtexec\s+report|tensorrt\s*exec\s+report' }
    )) {
    if ($text -match $pattern.regex) { $findings.Add([string]$pattern.id) | Out-Null }
  }

  return @($findings.ToArray() | Select-Object -Unique)
}

function New-OwnerPublicPublishResultSummary {
  param(
    [AllowNull()][object]$FieldMap,
    [AllowNull()][object]$GitHubActionsRunEvidence
  )

  [pscustomobject]@{
    publicPackageId = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "publicPackageId"
    publicPackageVersion = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "publicPackageVersion"
    publicPackageSource = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "publicPackageSource"
    publicPackageUrl = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "publicPackageUrl"
    publicPackageSha256 = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "publicPackageSha256"
    publicPackagePublishedAtUtc = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "publicPackagePublishedAtUtc"
    managedPackageId = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "managedPackageId"
    runtimePackageId = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "runtimePackageId"
    managedPackageUrl = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "managedPackageUrl"
    runtimePackageUrl = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "runtimePackageUrl"
    managedPackageSha256 = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "managedPackageSha256"
    runtimePackageSha256 = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "runtimePackageSha256"
    githubReleaseUrl = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "githubReleaseUrl"
    githubReleaseAssetUrl = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "githubReleaseAssetUrl"
    githubReleaseAssetSha256 = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "githubReleaseAssetSha256"
    packageManagedPackageSourceChannel = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "packageManagedPackageSourceChannel"
    packageSourceChannel = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "packageSourceChannel"
    ownerReviewer = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "ownerReviewer"
    ownerReviewTimestampUtc = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "ownerReviewTimestampUtc"
    ownerAuthorizationId = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "ownerAuthorizationId"
    ownerAuthorizationTimestampUtc = Get-OwnerPublicPublishFieldValue -FieldMap $FieldMap -Name "ownerAuthorizationTimestampUtc"
    sourceGitHubActionsRunEvidenceReady = [bool](Get-OwnerPropertyOrDefault -Object $GitHubActionsRunEvidence -Name "githubActionsRunEvidenceReady" -DefaultValue $false)
    sourceGitHubActionsRunId = [string](Get-OwnerPropertyOrDefault -Object $GitHubActionsRunEvidence -Name "runId" -DefaultValue "")
    sourceGitHubActionsRunUrl = [string](Get-OwnerPropertyOrDefault -Object $GitHubActionsRunEvidence -Name "runUrl" -DefaultValue "")
    sourceHeadSha = [string](Get-OwnerPropertyOrDefault -Object $GitHubActionsRunEvidence -Name "headSha" -DefaultValue "")
    sourceWorkflowRunLogSha256 = [string](Get-OwnerPropertyOrDefault -Object $GitHubActionsRunEvidence -Name "workflowRunLogSha256" -DefaultValue "")
    sourceArtifactManifestSha256 = [string](Get-OwnerPropertyOrDefault -Object $GitHubActionsRunEvidence -Name "artifactManifestSha256" -DefaultValue "")
  }
}
