using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace YoloVisionSample;

public sealed class YoloCapabilityEntry
{
    public YoloCapabilityEntry(
        YoloModelFamily family,
        YoloTaskType taskType,
        string familyAlias,
        string taskAlias,
        string decodePath,
        string auxiliaryMetadata,
        string evidenceLevel,
        bool supported,
        string supportState)
    {
        Family = family;
        TaskType = taskType;
        FamilyAlias = familyAlias ?? throw new ArgumentNullException(nameof(familyAlias));
        TaskAlias = taskAlias ?? throw new ArgumentNullException(nameof(taskAlias));
        DecodePath = decodePath ?? throw new ArgumentNullException(nameof(decodePath));
        AuxiliaryMetadata = auxiliaryMetadata ?? throw new ArgumentNullException(nameof(auxiliaryMetadata));
        EvidenceLevel = evidenceLevel ?? throw new ArgumentNullException(nameof(evidenceLevel));
        Supported = supported;
        SupportState = supportState ?? throw new ArgumentNullException(nameof(supportState));
    }

    public YoloModelFamily Family { get; }

    public YoloTaskType TaskType { get; }

    public string FamilyAlias { get; }

    public string TaskAlias { get; }

    public string DecodePath { get; }

    public string AuxiliaryMetadata { get; }

    public string EvidenceLevel { get; }

    public bool Supported { get; }

    public string SupportState { get; }
}

public static class YoloCapabilityMatrix
{
    private static readonly (YoloModelFamily Family, string Alias)[] Families =
    {
        (YoloModelFamily.Custom, "custom"),
        (YoloModelFamily.YoloV5, "v5"),
        (YoloModelFamily.YoloV6, "v6"),
        (YoloModelFamily.YoloV7, "v7"),
        (YoloModelFamily.YoloV8, "v8"),
        (YoloModelFamily.YoloV9, "v9"),
        (YoloModelFamily.YoloV10, "v10"),
        (YoloModelFamily.YoloV11, "v11"),
        (YoloModelFamily.YoloV26, "v26"),
        (YoloModelFamily.YoloX, "yolox")
    };

    private static readonly (YoloTaskType Task, string Alias, string DecodePath, string AuxiliaryMetadata, string EvidenceLevel)[] Tasks =
    {
        (YoloTaskType.Detection, "det", "single-output boxes with score filtering and class-aware/class-agnostic NMS", "none", "runtime-smoke-ready"),
        (YoloTaskType.Classification, "cls", "single-output raw/logits/probabilities decoder with strict class count and top-k", "classification score mode", "source-tree-real-model-runtime"),
        (YoloTaskType.Segmentation, "seg", "detection rows plus mask prototype composition", "mask coefficient count, prototype tensor role, optional auxiliary channel start/layout", "managed-metadata-ready"),
        (YoloTaskType.OrientedBoundingBox, "obb", "embedded or separate angle channels plus probabilistic-IoU rotated Fast-NMS", "angle unit and exact auxiliary start/layout for embedded output, or a separate angle tensor role", "source-tree-real-model-runtime"),
        (YoloTaskType.Pose, "pose", "detection rows plus keypoint tensor mapping", "keypoint count, keypoint stride, optional auxiliary layout", "managed-metadata-ready"),
        (YoloTaskType.SemanticSegmentation, "sem", "single-output semantic map decoder", "class count and semantic tensor role", "managed-smoke-ready")
    };

    public static IReadOnlyList<YoloCapabilityEntry> Entries { get; } = Families
        .SelectMany(static family => Tasks.Select(task => CreateEntry(family, task)))
        .ToArray();

    private static YoloCapabilityEntry CreateEntry(
        (YoloModelFamily Family, string Alias) family,
        (YoloTaskType Task, string Alias, string DecodePath, string AuxiliaryMetadata, string EvidenceLevel) task)
    {
        if (family.Family == YoloModelFamily.YoloX)
        {
            bool supported = task.Task == YoloTaskType.Detection;
            return new YoloCapabilityEntry(
                family.Family,
                task.Task,
                family.Alias,
                task.Alias,
                supported ? "YOLOX raw grid/stride transform plus score filtering and application-side NMS" : "unsupported: built-in YOLOX profile is detection-only",
                supported ? "strides 8,16,32; boxes-first raw output; objectness channel" : "not applicable",
                supported ? "source-tree-real-model-runtime-ready" : "unsupported-design-boundary",
                supported,
                supported ? "supported" : "unsupported-family-task");
        }

        if (family.Family == YoloModelFamily.YoloV10 && task.Task == YoloTaskType.Detection)
        {
            return new YoloCapabilityEntry(
                family.Family,
                task.Task,
                family.Alias,
                task.Alias,
                "YOLOv10 end-to-end [1,N,6] x1/y1/x2/y2/score/classId decoder without second NMS",
                "explicit --layout end2end, six-column order, class count, confidence threshold",
                "managed-end-to-end-smoke-ready",
                supported: true,
                supportState: "supported");
        }

        return new YoloCapabilityEntry(
            family.Family,
            task.Task,
            family.Alias,
            task.Alias,
            task.DecodePath,
            task.AuxiliaryMetadata,
            task.EvidenceLevel,
            supported: true,
            supportState: "supported");
    }

    public static string FormatConsoleTable()
    {
        string[] lines = Entries
            .Select(static entry =>
                string.Join(
                    " | ",
                    entry.FamilyAlias,
                    entry.TaskAlias,
                    entry.TaskType,
                    entry.SupportState,
                    entry.DecodePath,
                    entry.AuxiliaryMetadata,
                    entry.EvidenceLevel))
            .ToArray();

        return string.Join(
            Environment.NewLine,
            new[]
            {
                "YoloVision Capability Matrix",
                "family | task | task-name | support-state | decode-path | auxiliary-metadata | evidence-level"
            }.Concat(lines));
    }

    public static string FormatJson()
    {
        var payload = new
        {
            matrixId = "yolovision-capability-matrix",
            sample = "samples/YoloVision",
            matrixState = "managed-capability-surface",
            proofBoundary = "capability matrix only; not runtime proof; not real-model-runtime proof; not package-consumer-runtime proof; not post-publish proof",
            familyCount = Families.Length,
            taskCount = Tasks.Length,
            entryCount = Entries.Count,
            families = Families.Select(static item => item.Alias).ToArray(),
            tasks = Tasks.Select(static item => item.Alias).ToArray(),
            requiredOwnerEvidence = new[]
            {
                "model ONNX source URL, license, SHA256, and export command",
                "labels source/license/SHA256 and class count",
                "input image or preprocessed tensor source/license/SHA256",
                "TensorRtExec or OnnxToEngine build-only report plus SHA256",
                "YoloVision run log with YoloVision Passed=True",
                "sample-run-evidence record validated by owner-filled hashes and host metadata"
            },
            forbiddenProofSubstitutes = new[]
            {
                "YoloVision matrix",
                "TensorRtExec report",
                "OnnxToEngine report",
                "template",
                "dry-run",
                "build-only",
                "screenshot",
                "local feed",
                "ProjectReference",
                "direct .nupkg",
                "failedBlockerCount=0"
            },
            entries = Entries.Select(static entry => new
            {
                family = entry.FamilyAlias,
                task = entry.TaskAlias,
                familyName = entry.Family.ToString(),
                taskName = entry.TaskType.ToString(),
                decodePath = entry.DecodePath,
                auxiliaryMetadata = entry.AuxiliaryMetadata,
                evidenceLevel = entry.EvidenceLevel,
                supported = entry.Supported,
                supportState = entry.SupportState,
                canPromoteRealModelRuntime = false,
                canPromotePackageConsumerRuntime = false
            }).ToArray()
        };

        return JsonSerializer.Serialize(payload, new JsonSerializerOptions
        {
            WriteIndented = true
        });
    }
}
