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

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

function New-Field {
  param(
    [string]$Name,
    [string]$Type,
    [string]$ValidatorItemId,
    [string]$ProofRole,
    [bool]$Required = $true,
    [bool]$PlaceholderAllowed = $false
  )

  [pscustomobject]@{
    name = $Name
    required = $Required
    type = $Type
    placeholderAllowed = $PlaceholderAllowed
    validatorItemId = $ValidatorItemId
    proofRole = $ProofRole
  }
}

$fields = @(
  New-Field -Name "recordKind" -Type "string" -ValidatorItemId "record-kind" -ProofRole "must be package-consumer-runtime-proof-owner-input"
  New-Field -Name "sourceGitHubActionsRunEvidenceImportPath" -Type "file path" -ValidatorItemId "source-github-actions-run-evidence-import-present" -ProofRole "optional context pointing to imported GitHub Actions dry-run evidence; not proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "sourceGitHubActionsRunId" -Type "string" -ValidatorItemId "source-github-actions-run-evidence-import-present" -ProofRole "GitHub Actions run id used only as dry-run context; not proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "sourceGitHubActionsRunUrl" -Type "string" -ValidatorItemId "source-github-actions-run-evidence-import-present" -ProofRole "GitHub Actions run URL used only as dry-run context; not proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "sourceHeadSha" -Type "git SHA string" -ValidatorItemId "source-github-actions-run-evidence-import-present" -ProofRole "head SHA from dry-run import; not proof and not a runtime proof claim" -Required $false -PlaceholderAllowed $true
  New-Field -Name "packageDryRunArtifactPath" -Type "file path" -ValidatorItemId "source-github-actions-dry-run-pack-claim-ready" -ProofRole "dry-run managed package artifact path; not proof and must not be reused as public managedNupkgPath" -Required $false -PlaceholderAllowed $true
  New-Field -Name "packageDryRunManagedNupkgSha256" -Type "sha256 hex string" -ValidatorItemId "source-github-actions-dry-run-pack-claim-ready" -ProofRole "dry-run managed nupkg hash for comparison only; not proof and not the downloaded public package hash" -Required $false -PlaceholderAllowed $true
  New-Field -Name "packageDryRunCanClaimPack" -Type "boolean" -ValidatorItemId "source-github-actions-dry-run-pack-claim-ready" -ProofRole "true only when imported run can claim package-managed dry-run pack success; not proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "isDryRunOnly" -Type "boolean" -ValidatorItemId "dry-run-only-not-proof" -ProofRole "must remain true for imported dry-run context; dry-run is not proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "isPublishedPackageProof" -Type "boolean" -ValidatorItemId "published-package-proof-false" -ProofRole "must remain false; owner input template is not proof and not published package proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "isPackageConsumerRuntimeProof" -Type "boolean" -ValidatorItemId "package-consumer-runtime-proof-false" -ProofRole "must remain false; owner input template is not proof and not runtime proof" -Required $false -PlaceholderAllowed $true
  New-Field -Name "cleanExternalConsumerRoot" -Type "absolute directory path" -ValidatorItemId "clean-root-outside-repository" -ProofRole "clean consumer root outside this repository"
  New-Field -Name "consumerProjectPath" -Type "absolute or repository-relative .csproj path" -ValidatorItemId "consumer-project-exists" -ProofRole "external consumer project to scan and validate"
  New-Field -Name "publicPackageSource" -Type "string URL or public feed id" -ValidatorItemId "public-package-source-not-local" -ProofRole "public package source; local feeds are forbidden"
  New-Field -Name "managedPackageId" -Type "string" -ValidatorItemId "field-managedPackageId" -ProofRole "managed package identity"
  New-Field -Name "managedPackageVersion" -Type "SemVer string" -ValidatorItemId "field-managedPackageVersion" -ProofRole "managed package version restored by clean consumer"
  New-Field -Name "managedNupkgPath" -Type "file path" -ValidatorItemId "managed-nupkg-hash-match" -ProofRole "public managed nupkg hash evidence"
  New-Field -Name "managedNupkgSha256" -Type "sha256 hex string" -ValidatorItemId "managed-nupkg-sha256-format" -ProofRole "managed nupkg integrity"
  New-Field -Name "runtimePackageId" -Type "string" -ValidatorItemId "field-runtimePackageId" -ProofRole "runtime package identity"
  New-Field -Name "runtimePackageVersion" -Type "SemVer string" -ValidatorItemId "field-runtimePackageVersion" -ProofRole "runtime package version restored by clean consumer"
  New-Field -Name "runtimePackageKey" -Type "string" -ValidatorItemId "runtime-key-ready" -ProofRole "runtime package key used by smoke command"
  New-Field -Name "runtimeNupkgPath" -Type "file path" -ValidatorItemId "runtime-nupkg-hash-match" -ProofRole "public runtime nupkg hash evidence"
  New-Field -Name "runtimeNupkgSha256" -Type "sha256 hex string" -ValidatorItemId "runtime-nupkg-sha256-format" -ProofRole "runtime nupkg integrity"
  New-Field -Name "ownerName" -Type "string" -ValidatorItemId "field-ownerName" -ProofRole "human owner of external runtime proof"
  New-Field -Name "machineName" -Type "string" -ValidatorItemId "field-machineName" -ProofRole "host identity for reproducibility"
  New-Field -Name "hostOs" -Type "string" -ValidatorItemId "field-hostOs" -ProofRole "host OS evidence"
  New-Field -Name "hostArchitecture" -Type "string" -ValidatorItemId "field-hostArchitecture" -ProofRole "host architecture evidence"
  New-Field -Name "gpuName" -Type "string" -ValidatorItemId "field-gpuName" -ProofRole "GPU used for runtime smoke"
  New-Field -Name "cudaDriverVersion" -Type "string" -ValidatorItemId "field-cudaDriverVersion" -ProofRole "CUDA driver evidence"
  New-Field -Name "cudaDriverSupportedRuntime" -Type "string" -ValidatorItemId "field-cudaDriverSupportedRuntime" -ProofRole "driver/runtime compatibility"
  New-Field -Name "cudaRuntimeVersion" -Type "string" -ValidatorItemId "field-cudaRuntimeVersion" -ProofRole "CUDA runtime evidence"
  New-Field -Name "cudnnVersion" -Type "string" -ValidatorItemId "field-cudnnVersion" -ProofRole "cuDNN runtime evidence"
  New-Field -Name "tensorRtVersion" -Type "string" -ValidatorItemId "field-tensorRtVersion" -ProofRole "TensorRT runtime version evidence"
  New-Field -Name "tensorRtLine" -Type "string" -ValidatorItemId "field-tensorRtLine" -ProofRole "TensorRT line such as TRT10 or TRT11"
  New-Field -Name "restoreCommand" -Type "string" -ValidatorItemId "field-restoreCommand" -ProofRole "actual clean consumer restore command"
  New-Field -Name "buildCommand" -Type "string" -ValidatorItemId "field-buildCommand" -ProofRole "actual clean consumer build command"
  New-Field -Name "smokeCommand" -Type "string" -ValidatorItemId "smoke-command-runtime-key" -ProofRole "actual package consumer runtime smoke command"
  New-Field -Name "exitCode" -Type "integer string" -ValidatorItemId "exit-code-zero" -ProofRole "runtime smoke process exit code; must be 0"
  New-Field -Name "startedAtUtc" -Type "DateTimeOffset string" -ValidatorItemId "started-at-utc-parseable" -ProofRole "runtime smoke start timestamp"
  New-Field -Name "finishedAtUtc" -Type "DateTimeOffset string" -ValidatorItemId "finished-at-utc-parseable" -ProofRole "runtime smoke finish timestamp"
  New-Field -Name "dependencyProbeStatus" -Type "enum string" -ValidatorItemId "dependency-probe-status-passed" -ProofRole "dependency probe status; must be passed or compatible-host-passed"
  New-Field -Name "smokeStatus" -Type "enum string" -ValidatorItemId "smoke-status-passed" -ProofRole "runtime smoke status; must be passed"
  New-Field -Name "nativeAssetsCopied" -Type "boolean string" -ValidatorItemId "native-assets-copied-true" -ProofRole "runtime native asset copy confirmation"
  New-Field -Name "smokeLogPath" -Type "file path" -ValidatorItemId "smoke-log-hash-match" -ProofRole "runtime smoke log evidence"
  New-Field -Name "smokeLogSha256" -Type "sha256 hex string" -ValidatorItemId "smoke-log-sha256-format" -ProofRole "runtime smoke log integrity"
  New-Field -Name "stdoutSummary" -Type "string" -ValidatorItemId "field-stdoutSummary" -ProofRole "stdout summary from real run"
  New-Field -Name "stderrSummary" -Type "string" -ValidatorItemId "field-stderrSummary" -ProofRole "stderr summary from real run"
  New-Field -Name "failureDiagnostic" -Type "string" -ValidatorItemId "field-failureDiagnostic" -ProofRole "failure detail or explicit empty diagnostic"
  New-Field -Name "performsPublish" -Type "boolean" -ValidatorItemId "no-side-effects" -ProofRole "must remain false; owner input cannot publish"
  New-Field -Name "canPublishPublicly" -Type "boolean" -ValidatorItemId "no-side-effects" -ProofRole "must remain false; owner input cannot approve publication"
  New-Field -Name "canCloseReleaseIssue" -Type "boolean" -ValidatorItemId "no-side-effects" -ProofRole "must remain false; owner input cannot close release issue"
  New-Field -Name "canPromoteProof" -Type "boolean" -ValidatorItemId "no-side-effects" -ProofRole "must remain false; import/validation decide only readiness, not proof promotion"
)

$forbiddenSubstitutes = @(
  [pscustomobject]@{ id = "local-feed"; label = "local feed"; reason = "A local folder/feed is dependency-probe evidence, not public package-consumer runtime proof." },
  [pscustomobject]@{ id = "project-reference"; label = "ProjectReference"; reason = "ProjectReference proves source-tree compatibility, not public package consumption." },
  [pscustomobject]@{ id = "direct-nupkg"; label = "direct .nupkg"; reason = "Direct file references bypass public package restore semantics." },
  [pscustomobject]@{ id = "repository-path-leakage"; label = "repository path leakage"; reason = "Clean consumer proof must run outside the repository and cannot depend on source paths." },
  [pscustomobject]@{ id = "build-only"; label = "build-only"; reason = "Build success does not prove TensorRT runtime execution." },
  [pscustomobject]@{ id = "dry-run"; label = "dry-run"; reason = "Dry-run output does not deserialize, bind, enqueue, or validate runtime output." },
  [pscustomobject]@{ id = "github-actions-dry-run-nupkg"; label = "GitHub Actions dry-run .nupkg"; reason = "The dry-run nupkg SHA256 is comparison context only and is not the downloaded public package SHA256." },
  [pscustomobject]@{ id = "dashboard"; label = "dashboard"; reason = "Dashboards summarize status but cannot replace machine-verifiable clean consumer runtime proof." },
  [pscustomobject]@{ id = "template-placeholder"; label = "template placeholder"; reason = "Placeholder/template values are owner-action-required, not proof." },
  [pscustomobject]@{ id = "gui-screenshot"; label = "GUI screenshot"; reason = "Screenshots are not machine-verifiable runtime proof." },
  [pscustomobject]@{ id = "tensorrtexec-build-report-only"; label = "TensorRtExec build report only"; reason = "TensorRtExec build/read-only diagnostics cannot promote package-consumer runtime proof." }
)

$schema = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-owner-input-schema"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sourceTemplate = "artifacts/final-release/package-consumer-runtime-proof-owner-input.template.json"
  validationScript = "eng/Test-PackageConsumerRuntimeProofOwnerInput.ps1"
  importScript = "eng/Import-PackageConsumerRuntimeProofOwnerInput.ps1"
  proofLineId = "package-consumer-runtime"
  fieldCount = $fields.Count
  requiredFieldCount = @($fields | Where-Object { $_.required }).Count
  placeholderAllowedFieldCount = @($fields | Where-Object { $_.placeholderAllowed }).Count
  fields = @($fields)
  forbiddenSubstitutes = @($forbiddenSubstitutes)
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPromoteProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  safetyBoundary = "Schema artifact only. It documents owner input fields and forbidden substitutes; it does not publish packages, close release issues, import owner input, or promote package-consumer runtime proof. GitHub Actions package dry-run context can prefill owner work items, but its nupkg SHA256 is not the downloaded public package SHA256 and cannot satisfy package-consumer runtime proof."
}

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input.schema.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input.schema.md"

$schema | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$fieldRows = $schema.fields | ForEach-Object {
  "| ``$($_.name)`` | ``$($_.required)`` | ``$($_.type)`` | ``$($_.placeholderAllowed)`` | ``$($_.validatorItemId)`` | $($_.proofRole.Replace("|", "\|")) |"
}

$substituteRows = $schema.forbiddenSubstitutes | ForEach-Object {
  "| ``$($_.id)`` | $($_.label.Replace("|", "\|")) | $($_.reason.Replace("|", "\|")) |"
}

$markdown = @"
# Package Consumer Runtime Proof Owner Input Schema

生成时间：$($schema.generatedAtUtc)

## 用途

该 schema 产物记录 Owner 回填 `package-consumer-runtime` 证据所需字段、对应 validator item 和 proof role。它只定义输入契约，不执行发布、不导入 owner input、不关闭 release issue，也不会把任何 template、local feed、ProjectReference、direct `.nupkg`、dry-run、dashboard 或 build-only 输出晋级为 runtime proof。`packageDryRunManagedNupkgSha256` 仅用于对照 GitHub Actions dry-run pack，不是 Owner 从 public feed 下载的 package hash。

| 项目 | 当前值 |
|---|---|
| proofLineId | ``$($schema.proofLineId)`` |
| fieldCount | ``$($schema.fieldCount)`` |
| requiredFieldCount | ``$($schema.requiredFieldCount)`` |
| placeholderAllowedFieldCount | ``$($schema.placeholderAllowedFieldCount)`` |
| validationScript | ``$($schema.validationScript)`` |
| importScript | ``$($schema.importScript)`` |
| canPromoteRuntimeProof | ``$($schema.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($schema.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($schema.canCloseReleaseIssue)`` |

## Fields

| Field | Required | Type | Placeholder Allowed | Validator Item | Proof Role |
|---|---|---|---|---|---|
$($fieldRows -join "`r`n")

## Forbidden Substitutes

| ID | Substitute | Reason |
|---|---|---|
$($substituteRows -join "`r`n")

## Safety Boundary

$($schema.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof owner input schema written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "FieldCount=$($schema.fieldCount) PlaceholderAllowedFieldCount=$($schema.placeholderAllowedFieldCount) CanPromoteRuntimeProof=False"
