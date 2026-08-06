using System.Diagnostics;
using System.Text.Json;
using Xunit;

namespace JYPPX.ProjectQuality.Tests;

public sealed class YoloVisionRealAssetCandidateValidatorTests
{
    [Fact]
    public void YoloVisionRealAssetCandidateValidatorAcceptsTemplatesButDoesNotPromoteProof()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetCandidate.ps1"), "-Strict");

        string validationPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "yolovision-real-asset-candidate-validation.json");

        Assert.True(File.Exists(validationPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-real-asset-candidate-validation", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("validationState").GetString());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());

        JsonElement.ArrayEnumerator records = root.GetProperty("records").EnumerateArray();
        JsonElement[] recordArray = records.ToArray();
        Assert.Equal(6, recordArray.Length);
        Assert.Equal(new[] { "cls", "det", "obb", "pose", "seg", "sem" }, recordArray.Select(record => record.GetProperty("task").GetString()).OrderBy(static task => task).ToArray());

        foreach (JsonElement record in recordArray)
        {
            Assert.Equal("YoloVision", record.GetProperty("sampleName").GetString());
            Assert.Equal("YOLOv8", record.GetProperty("family").GetString());
            string task = record.GetProperty("task").GetString()!;
            Assert.Contains(task, new[] { "det", "seg", "pose", "obb", "cls", "sem" });
            Assert.Equal("owner-action-required", record.GetProperty("runtimeProofState").GetString());
            Assert.Equal("template-only", record.GetProperty("proofClassification").GetString());
            Assert.Equal("owner-action-required", record.GetProperty("validationState").GetString());
            Assert.False(record.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(record.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

            AssertValidationContains(record, "sample-name-yolovision", passed: true);
            AssertValidationContains(record, "family-yolov8", passed: true);
            AssertValidationContains(record, "task-supported", passed: true);
            AssertValidationContains(record, "package-consumer-never-promoted", passed: true);
            AssertValidationContains(record, "template-cannot-promote-real-model-runtime", passed: true);
            AssertValidationContains(record, "expected-evidence-passed-line", passed: true);
            AssertValidationContains(record, "required-hash-listed-modelSha256", passed: true);
            AssertValidationContains(record, "required-hash-listed-labelsSha256", passed: true);
            AssertValidationContains(record, "required-hash-listed-imageSha256", passed: true);
            AssertValidationContains(record, "required-hash-listed-preprocessedTensorSha256", passed: true);
            AssertValidationContains(record, "required-hash-listed-runLogSha256", passed: true);

            if (task == "pose")
            {
                AssertValidationContains(record, "pose-keypoint-count-present", passed: true);
                AssertValidationContains(record, "pose-keypoint-layout-present", passed: true);
                AssertValidationContains(record, "pose-score-field-present", passed: true);
            }
            else if (task == "obb")
            {
                AssertValidationContains(record, "obb-angle-unit-present", passed: true);
                AssertValidationContains(record, "obb-rotated-box-layout-present", passed: true);
                AssertValidationContains(record, "obb-coordinate-space-present", passed: true);
            }
            else if (task == "cls")
            {
                AssertValidationContains(record, "cls-topk-present", passed: true);
                AssertValidationContains(record, "cls-score-field-present", passed: true);
                AssertValidationContains(record, "cls-labels-required-present", passed: true);
            }
            else if (task == "sem")
            {
                AssertValidationContains(record, "sem-map-shape-present", passed: true);
                AssertValidationContains(record, "sem-class-map-layout-present", passed: true);
                AssertValidationContains(record, "sem-palette-required-present", passed: true);
            }
        }
    }

    [Fact]
    public void YoloVisionValidatorDocumentationAndScriptCaptureOwnerBackfillBoundaries()
    {
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetCandidate.ps1"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-real-asset-owner-backfill-validator.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("function Test-Sha256", script, StringComparison.Ordinal);
        Assert.Contains("package-consumer-never-promoted", script, StringComparison.Ordinal);
        Assert.Contains("real-runtime-promotion-requires-all-hashes", script, StringComparison.Ordinal);
        Assert.Contains("real-runtime-promotion-requires-stdout-stderr", script, StringComparison.Ordinal);
        Assert.Contains("task-supported", script, StringComparison.Ordinal);
        Assert.Contains("pose-keypoint-count-present", script, StringComparison.Ordinal);
        Assert.Contains("obb-rotated-box-layout-present", script, StringComparison.Ordinal);
        Assert.Contains("cls-topk-present", script, StringComparison.Ordinal);
        Assert.Contains("sem-map-shape-present", script, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", script, StringComparison.Ordinal);
        Assert.Contains("performsPublish = $false", script, StringComparison.Ordinal);

        Assert.Contains("articles/zh-cn/yolovision-real-asset-owner-backfill-validator.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-real-asset-owner-backfill-validator.md", docsToc, StringComparison.Ordinal);
        Assert.Contains("owner-action-required", article, StringComparison.Ordinal);
        Assert.Contains("64 位 SHA256", article, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", article, StringComparison.Ordinal);
        Assert.Contains("不能晋级 package-consumer-runtime", article, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionRealAssetCandidate.ps1", article, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionOwnerBackfillPackValidatorKeepsTemplateBlockedUntilRealOwnerEvidence()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetOwnerBackfillPack.ps1"), "-Strict");

        string validationPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "yolovision-real-asset-owner-backfill-pack-validation.json");

        Assert.True(File.Exists(validationPath));

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement root = document.RootElement;

        Assert.Equal("yolovision-real-asset-owner-backfill-pack-validation", root.GetProperty("recordKind").GetString());
        Assert.False(root.GetProperty("performsPublish").GetBoolean());
        Assert.False(root.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(root.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(root.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Equal("owner-action-required", root.GetProperty("validationState").GetString());
        Assert.Equal(0, root.GetProperty("failedBlockerCount").GetInt32());
        Assert.Equal("applications/YoloVision/yolovision-task-output-contract.json", root.GetProperty("contractPath").GetString());
        Assert.Equal(6, root.GetProperty("contractTaskCount").GetInt32());
        Assert.Equal(6, root.GetProperty("articleCaseTaskCount").GetInt32());
        foreach (string family in new[] { "yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom" })
        {
            AssertValidationContains(root, "contract-family-" + family, passed: true);
        }

        JsonElement[] records = root.GetProperty("records").EnumerateArray().ToArray();
        Assert.Equal(6, records.Length);
        Assert.Equal(new[] { "cls", "det", "obb", "pose", "seg", "sem" }, records.Select(static record => record.GetProperty("task").GetString()).OrderBy(static task => task).ToArray());

        foreach (JsonElement record in records)
        {
            string task = record.GetProperty("task").GetString()!;
            Assert.Equal("owner-action-required", record.GetProperty("validationState").GetString());
            Assert.False(record.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(record.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            AssertValidationContains(record, "case-no-real-promotion", passed: true);
            AssertValidationContains(record, "case-no-package-promotion", passed: true);
            AssertValidationContains(record, "case-task-in-contract", passed: true);
            Assert.Equal(task == "sem" ? "custom" : "v8", record.GetProperty("family").GetString());
            Assert.Equal(task == "sem" ? "custom" : "yolov8", record.GetProperty("normalizedFamily").GetString());
            AssertValidationContains(record, "case-family-in-contract", passed: true);
            AssertValidationContains(record, "case-family-command-match", passed: true);
            AssertValidationContains(record, "case-preflight-family-match", passed: true);
            AssertValidationContains(record, "article-file-exists", passed: true);
            AssertValidationContains(record, "article-entrypoint-in-contract", passed: true);
            AssertValidationContains(record, "article-case-family-task-match", passed: true);
            AssertValidationContains(record, "contract-required-metadata-present", passed: true);
            AssertValidationContains(record, "contract-profile-hint-present", passed: true);
            AssertValidationContains(record, "tensorrtexec-profile-hint-aligned", passed: true);
            AssertValidationContains(record, "tensorrtexec-build-command", passed: true);
            AssertValidationContains(record, "tensorrtexec-build-only", passed: true);
            AssertValidationContains(record, "yolovision-run-command", passed: true);
            AssertValidationContains(record, "preflight-command", passed: true);
            AssertValidationContains(record, "preflight-schema-version", passed: true);
            AssertValidationContains(record, "preflight-proof-classification", passed: true);
            AssertValidationContains(record, "preflight-execution-disabled", passed: true);
            AssertValidationContains(record, "preflight-boundary-disabled", passed: true);
            AssertValidationContains(record, "preflight-report-schema", passed: true);
            AssertValidationContains(record, "preflight-report-execution-disabled", passed: true);
            AssertValidationContains(record, "preflight-report-boundary-disabled", passed: true);
            AssertValidationContains(record, "expected-yolovision-passed", passed: true);
            AssertValidationContains(record, "run-log-sha256-required-or-real", passed: true);
            AssertValidationContains(record, "output-json-sha256-required-or-real", passed: true);
            AssertValidationContains(record, "owner-review-does-not-accept-template", passed: true);
        }
    }

    [Fact]
    public void YoloVisionOwnerBackfillPackDocsAndScriptsAreLinkedAndPreserveProofBoundaries()
    {
        string pack = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "yolovision-real-asset-owner-backfill-pack.json"));
        string script = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetOwnerBackfillPack.ps1"));
        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerBackfillPack.ps1"));
        string article = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "articles", "zh-cn", "yolovision-real-asset-owner-backfill-pack.md"));
        string assetsReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "samples", "assets", "README.md"));
        string yoloReadme = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "applications", "YoloVision", "README.md"));
        string docsIndex = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "index.md"));
        string docsToc = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "docs", "toc.yml"));

        Assert.Contains("yolovision-real-asset-owner-backfill-pack", pack, StringComparison.Ordinal);
        Assert.Contains("model.sourceUrl", article, StringComparison.Ordinal);
        Assert.Contains("TensorRtExec build-only", article, StringComparison.Ordinal);
        Assert.Contains("YoloVision Passed=True", article, StringComparison.Ordinal);
        Assert.Contains("64 位十六进制字符串", article, StringComparison.Ordinal);
        Assert.Contains("不能由本包替代", article, StringComparison.Ordinal);
        Assert.Contains("Test-YoloVisionRealAssetOwnerBackfillPack.ps1", article, StringComparison.Ordinal);
        Assert.Contains("function Test-Sha256", script, StringComparison.Ordinal);
        Assert.Contains("yolovision-task-output-contract.json", script, StringComparison.Ordinal);
        Assert.Contains("case-task-in-contract", script, StringComparison.Ordinal);
        Assert.Contains("Normalize-YoloFamilyAlias", script, StringComparison.Ordinal);
        Assert.Contains("Add-ValidationItem $items (\"contract-family-\" + $family)", script, StringComparison.Ordinal);
        Assert.Contains("article-entrypoint-in-contract", script, StringComparison.Ordinal);
        Assert.Contains("case-preflight-family-match", script, StringComparison.Ordinal);
        Assert.Contains("contract-required-metadata-", script, StringComparison.Ordinal);
        Assert.Contains("tensorrtexec-profile-hint-aligned", script, StringComparison.Ordinal);
        Assert.Contains("preflight-report-schema", script, StringComparison.Ordinal);
        Assert.Contains("preflight-report-boundary-disabled", script, StringComparison.Ordinal);
        Assert.Contains("owner-review-does-not-accept-template", script, StringComparison.Ordinal);
        Assert.Contains("canPromotePackageConsumerRuntime = $false", script, StringComparison.Ordinal);
        Assert.Contains("yolovision-real-asset-owner-backfill-sample-run-evidence.template.json", exporter, StringComparison.Ordinal);
        Assert.Contains("projection-aligned", exporter, StringComparison.Ordinal);
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", exporter, StringComparison.Ordinal);
        Assert.Contains("yolovision-real-asset-owner-backfill-pack.json", assetsReadme, StringComparison.Ordinal);
        Assert.Contains("Export-YoloVisionRealAssetOwnerBackfillPack.ps1", assetsReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-real-asset-owner-backfill-sample-run-evidence.template.json", assetsReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-real-asset-owner-backfill-pack.json", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("yolovision-real-asset-owner-backfill-sample-run-evidence.template.json", yoloReadme, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-real-asset-owner-backfill-pack.md", docsIndex, StringComparison.Ordinal);
        Assert.Contains("articles/zh-cn/yolovision-real-asset-owner-backfill-pack.md", docsToc, StringComparison.Ordinal);
        Assert.DoesNotContain("YoloDet", article + pack + assetsReadme + yoloReadme, StringComparison.Ordinal);
    }

    [Fact]
    public void YoloVisionOwnerBackfillExporterCreatesSixTaskSampleRunEvidenceTemplate()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerBackfillPack.ps1"));

        string templatePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "user-acceptance",
            "yolovision-real-asset-owner-backfill-sample-run-evidence.template.json");
        string templateMarkdownPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "user-acceptance",
            "yolovision-real-asset-owner-backfill-sample-run-evidence.template.md");
        string reportPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "yolovision",
            "yolovision-real-asset-owner-backfill-pack-projection-report.json");
        string generatedPath = Path.Combine(
            RepositoryPaths.Root,
            "samples",
            "assets",
            "yolovision-real-asset-owner-backfill-pack.generated.json");

        Assert.True(File.Exists(templatePath));
        Assert.True(File.Exists(templateMarkdownPath));
        Assert.True(File.Exists(reportPath));
        Assert.True(File.Exists(generatedPath));

        using JsonDocument templateDocument = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("yolovision-real-asset-owner-backfill-sample-run-evidence-template", template.GetProperty("recordKind").GetString());
        Assert.True(template.GetProperty("templateOnly").GetBoolean());
        Assert.Equal("template-only", template.GetProperty("proofClassification").GetString());
        Assert.Equal("owner-action-required", template.GetProperty("validationState").GetString());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(template.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("not real-model-runtime proof", template.GetProperty("proofBoundary").GetString(), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", template.GetProperty("validator").GetString(), StringComparison.Ordinal);
        Assert.Equal("yolovision-preflight.v1", template.GetProperty("preflightContract").GetProperty("schemaVersion").GetString());
        Assert.Equal("precheck", template.GetProperty("preflightContract").GetProperty("proofClassification").GetString());
        Assert.Equal(7, template.GetProperty("requiredPreflightEvidence").GetArrayLength());

        JsonElement[] cases = template.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(6, cases.Length);
        Assert.Contains(cases, static item => item.GetProperty("task").GetString() == "sem");
        Assert.Equal(new[] { "cls", "det", "obb", "pose", "seg", "sem" }, cases.Select(static item => item.GetProperty("task").GetString()).OrderBy(static task => task).ToArray());
        Assert.Contains("six-task", template.GetProperty("templateName").GetString(), StringComparison.Ordinal);

        foreach (JsonElement item in cases)
        {
            Assert.Equal("YoloVision", item.GetProperty("sampleName").GetString());
            Assert.True(item.GetProperty("templateOnly").GetBoolean());
            Assert.Equal("template-only", item.GetProperty("proofClassification").GetString());
            Assert.Equal("owner-action-required", item.GetProperty("state").GetString());
            Assert.False(item.GetProperty("isSmokePassed").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.Equal("owner-required", item.GetProperty("model").GetProperty("sha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("model").GetProperty("onnxSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("labels").GetProperty("sha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("input").GetProperty("imageSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("input").GetProperty("preprocessedTensorSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("tensorRtExec").GetProperty("reportSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("tensorRtExec").GetProperty("engineSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("yoloVisionPreflight").GetProperty("reportSha256").GetString());
            Assert.Equal("yolovision-preflight.v1", item.GetProperty("yoloVisionPreflight").GetProperty("schemaVersion").GetString());
            Assert.Equal("precheck", item.GetProperty("yoloVisionPreflight").GetProperty("proofClassification").GetString());
            Assert.False(item.GetProperty("yoloVisionPreflight").GetProperty("execution").GetProperty("engineBuildInvoked").GetBoolean());
            Assert.False(item.GetProperty("yoloVisionPreflight").GetProperty("boundary").GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.Equal("owner-required", item.GetProperty("sampleRunLogSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("outputJsonSha256").GetString());
            Assert.Contains("TensorRtExec report is build/report evidence only", item.GetProperty("tensorRtExec").GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);
            Assert.Contains(item.GetProperty("expectedEvidenceLines").EnumerateArray(), static line => line.GetString() == "YoloVision Passed=True");
        }

        using JsonDocument reportDocument = JsonDocument.Parse(File.ReadAllText(reportPath));
        JsonElement report = reportDocument.RootElement;
        Assert.Equal("yolovision-real-asset-owner-backfill-pack-projection-report", report.GetProperty("recordKind").GetString());
        Assert.Equal("projection-aligned", report.GetProperty("validationState").GetString());
        Assert.Equal(0, report.GetProperty("failedCount").GetInt32());
        Assert.False(report.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(report.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());

        string markdown = File.ReadAllText(templateMarkdownPath);
        Assert.Contains("yolov8n-det", markdown, StringComparison.Ordinal);
        Assert.Contains("yolov8n-cls", markdown, StringComparison.Ordinal);
        Assert.Contains("yolov8n-sem", markdown, StringComparison.Ordinal);
        Assert.Contains("not real-model-runtime proof", markdown, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void YoloVisionOwnerProofInputTemplateValidatorAndImporterKeepOwnerActionBoundary()
    {
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerBackfillPack.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1"));
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetOwnerProofInput.ps1"), "-Strict");
        RunPowerShell(Path.Combine(RepositoryPaths.Root, "eng", "Import-YoloVisionRealAssetOwnerProofInput.ps1"));

        string templatePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "user-acceptance",
            "yolovision-real-asset-owner-proof-input.template.json");
        string validationPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "user-acceptance",
            "yolovision-real-asset-owner-proof-input-validation.json");
        string importReportPath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "user-acceptance",
            "yolovision-real-asset-owner-proof-import-report.json");
        string candidatePath = Path.Combine(
            RepositoryPaths.Root,
            "artifacts",
            "user-acceptance",
            "yolovision-real-asset-owner-sample-run-evidence.candidate.json");

        Assert.True(File.Exists(templatePath));
        Assert.True(File.Exists(validationPath));
        Assert.True(File.Exists(importReportPath));
        Assert.True(File.Exists(candidatePath));

        using JsonDocument templateDocument = JsonDocument.Parse(File.ReadAllText(templatePath));
        JsonElement template = templateDocument.RootElement;
        Assert.Equal("yolovision-real-asset-owner-proof-input", template.GetProperty("recordKind").GetString());
        Assert.Equal("template-owner-input-required", template.GetProperty("ownerInputState").GetString());
        Assert.Equal("template-only", template.GetProperty("proofClassification").GetString());
        Assert.False(template.GetProperty("performsPublish").GetBoolean());
        Assert.False(template.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(template.GetProperty("canPromoteRealModelRuntime").GetBoolean());
        Assert.False(template.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.False(template.GetProperty("canCloseReleaseIssue").GetBoolean());
        Assert.Contains("Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog", template.GetProperty("sampleRunEvidenceValidator").GetString(), StringComparison.Ordinal);
        Assert.Contains(template.GetProperty("validationRules").EnumerateArray(), static item =>
            item.GetString()!.Contains("Package-consumer-runtime belongs to release proof records", StringComparison.Ordinal));

        JsonElement requiredGlobalEvidence = template.GetProperty("requiredGlobalEvidence");
        JsonElement hostMetadata = requiredGlobalEvidence.GetProperty("hostMetadata");
        foreach (string field in new[]
        {
            "hostOs",
            "hostMachineId",
            "osArchitecture",
            "gpuName",
            "gpuComputeCapability",
            "driverVersion",
            "cudaDriverVersion",
            "cudaRuntimeVersion",
            "tensorRtVersion",
            "tensorRtLine",
            "cudnnVersion"
        })
        {
            Assert.Equal("owner-required", hostMetadata.GetProperty(field).GetString());
        }

        JsonElement packageMetadata = requiredGlobalEvidence.GetProperty("packageMetadata");
        foreach (string field in new[]
        {
            "packageSource",
            "packageChannel",
            "runtimePackageVersion",
            "runtimePackageKey",
            "managedPackageSha256",
            "nativeBridgeSha256",
            "runtimePackageSha256"
        })
        {
            Assert.Equal("owner-required", packageMetadata.GetProperty(field).GetString());
        }

        JsonElement globalOwnerReview = requiredGlobalEvidence.GetProperty("ownerReview");
        foreach (string field in new[] { "ownerReviewer", "ownerReviewedAtUtc", "ownerAcceptanceDecision", "ownerAcceptanceNotes" })
        {
            Assert.Equal("owner-required", globalOwnerReview.GetProperty(field).GetString());
        }

        JsonElement[] templateCases = template.GetProperty("cases").EnumerateArray().ToArray();
        Assert.Equal(6, templateCases.Length);
        Assert.Contains(templateCases, static item => item.GetProperty("task").GetString() == "sem");
        foreach (JsonElement item in templateCases)
        {
            Assert.False(item.GetProperty("isSmokePassed").GetBoolean());
            Assert.False(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.Equal("owner-required", item.GetProperty("model").GetProperty("sha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("model").GetProperty("onnxSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("input").GetProperty("preprocessedTensorElementCount").GetString());
            Assert.Equal("owner-required", item.GetProperty("tensorRtExec").GetProperty("stdoutLogSha256").GetString());
            Assert.Equal("owner-required-or-no-stderr", item.GetProperty("tensorRtExec").GetProperty("stderrLogSha256").GetString());
            Assert.Equal("owner-required", item.GetProperty("yoloVision").GetProperty("stdoutLogSha256").GetString());
            Assert.Equal("owner-required-or-no-stderr", item.GetProperty("yoloVision").GetProperty("stderrLogSha256").GetString());
            Assert.Equal("near-ready-owner-evidence-missing", item.GetProperty("articleEvidence").GetProperty("articleStatus").GetString());
            Assert.Contains(item.GetProperty("yoloVision").GetProperty("expectedEvidenceLines").EnumerateArray(), static line => line.GetString() == "YoloVision Passed=True");
        }

        using JsonDocument validationDocument = JsonDocument.Parse(File.ReadAllText(validationPath));
        JsonElement validation = validationDocument.RootElement;
        Assert.Equal("yolovision-real-asset-owner-proof-input-validation", validation.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", validation.GetProperty("validationState").GetString());
        Assert.Equal(0, validation.GetProperty("failedBlockerCount").GetInt32());
        Assert.True(validation.GetProperty("ownerActionRequiredCount").GetInt32() >= 1);
        Assert.False(validation.GetProperty("candidateReadyForRealModelRuntime").GetBoolean());
        Assert.False(validation.GetProperty("performsPublish").GetBoolean());
        Assert.False(validation.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(validation.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        AssertValidationContains(validation, "global-host-hostOs", passed: false);
        AssertValidationContains(validation, "global-package-managedPackageSha256", passed: false);
        AssertValidationContains(validation, "global-owner-ownerAcceptanceDecision", passed: false);
        AssertValidationContains(validation, "case-yolov8n-det-model-sha256", passed: false);
        AssertValidationContains(validation, "case-yolov8n-det-trtexec-stdout-log-sha256", passed: false);
        AssertValidationContains(validation, "case-yolov8n-det-yolovision-stdout-log-sha256", passed: false);
        AssertValidationContains(validation, "case-yolov8n-det-expected-passed", passed: true);
        AssertValidationContains(validation, "case-yolov8n-det-no-package-promotion", passed: true);

        using JsonDocument importDocument = JsonDocument.Parse(File.ReadAllText(importReportPath));
        JsonElement import = importDocument.RootElement;
        Assert.Equal("yolovision-real-asset-owner-proof-import-report", import.GetProperty("recordKind").GetString());
        Assert.Equal("owner-action-required", import.GetProperty("validationState").GetString());
        Assert.False(import.GetProperty("candidateReadyForRealModelRuntime").GetBoolean());
        Assert.False(import.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Contains("never package-consumer-runtime proof", import.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);

        using JsonDocument candidateDocument = JsonDocument.Parse(File.ReadAllText(candidatePath));
        JsonElement candidate = candidateDocument.RootElement;
        Assert.Equal("yolovision-real-asset-owner-sample-run-evidence-candidate", candidate.GetProperty("recordKind").GetString());
        Assert.False(candidate.GetProperty("performsPublish").GetBoolean());
        Assert.False(candidate.GetProperty("canPublishPublicly").GetBoolean());
        Assert.False(candidate.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
        Assert.Equal(6, candidate.GetProperty("cases").GetArrayLength());
        Assert.All(candidate.GetProperty("cases").EnumerateArray(), item =>
        {
            Assert.Equal("sample-run-evidence-record", item.GetProperty("recordKind").GetString());
            Assert.Equal("real-model-runtime", item.GetProperty("proofClassification").GetString());
            Assert.False(item.GetProperty("canPromoteRealModelRuntime").GetBoolean());
            Assert.False(item.GetProperty("canPromotePackageConsumerRuntime").GetBoolean());
            Assert.Contains("never package-consumer-runtime proof", item.GetProperty("proofBoundary").GetString(), StringComparison.Ordinal);
        });

        string exporter = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Export-YoloVisionRealAssetOwnerProofInputTemplate.ps1"));
        string validator = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Test-YoloVisionRealAssetOwnerProofInput.ps1"));
        string importer = File.ReadAllText(Path.Combine(RepositoryPaths.Root, "eng", "Import-YoloVisionRealAssetOwnerProofInput.ps1"));
        Assert.Contains("forbiddenSubstitutes", exporter, StringComparison.Ordinal);
        Assert.Contains("candidateReadyForRealModelRuntime", validator, StringComparison.Ordinal);
        Assert.Contains("Package-consumer-runtime belongs to release proof records", exporter, StringComparison.Ordinal);
        Assert.Contains("not package-consumer-runtime proof", validator, StringComparison.Ordinal);
        Assert.Contains("never package-consumer-runtime proof", importer, StringComparison.Ordinal);
    }

    private static void AssertValidationContains(JsonElement record, string id, bool passed)
    {
        Assert.Contains(record.GetProperty("validationItems").EnumerateArray(), item =>
            item.GetProperty("id").GetString() == id &&
            item.GetProperty("passed").GetBoolean() == passed);
    }

    private static string RunPowerShell(string scriptPath, params string[] arguments)
    {
        using Process process = new();
        process.StartInfo.FileName = "pwsh";
        process.StartInfo.ArgumentList.Add("-NoProfile");
        process.StartInfo.ArgumentList.Add("-ExecutionPolicy");
        process.StartInfo.ArgumentList.Add("Bypass");
        process.StartInfo.ArgumentList.Add("-File");
        process.StartInfo.ArgumentList.Add(scriptPath);
        foreach (string argument in arguments)
        {
            process.StartInfo.ArgumentList.Add(argument);
        }

        process.StartInfo.WorkingDirectory = RepositoryPaths.Root;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.UseShellExecute = false;

        process.Start();
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        Assert.True(process.ExitCode == 0, $"Command failed: {scriptPath}{Environment.NewLine}{stdout}{Environment.NewLine}{stderr}");
        return stdout;
    }
}
