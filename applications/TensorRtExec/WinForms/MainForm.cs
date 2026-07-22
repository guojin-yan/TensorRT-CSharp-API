using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Windows.Forms;
using TensorRtExecApp.Core;

namespace TensorRtExecApp.WinForms;

public sealed class MainForm : Form
{
    private readonly TextBox _onnxPath = new TextBox();
    private readonly TextBox _enginePath = new TextBox();
    private readonly TextBox _loadEnginePath = new TextBox();
    private readonly TextBox _refitFromOnnxPath = new TextBox();
    private readonly TextBox _saveRefittedEnginePath = new TextBox();
    private readonly ComboBox _tensorRtLine = new ComboBox();
    private readonly CheckBox _fp16 = new CheckBox();
    private readonly CheckBox _int8 = new CheckBox();
    private readonly CheckBox _bf16 = new CheckBox();
    private readonly CheckBox _fp8 = new CheckBox();
    private readonly CheckBox _best = new CheckBox();
    private readonly CheckBox _tf32 = new CheckBox();
    private readonly NumericUpDown _workspace = new NumericUpDown();
    private readonly TextBox _minShapes = new TextBox();
    private readonly TextBox _optShapes = new TextBox();
    private readonly TextBox _maxShapes = new TextBox();
    private readonly TextBox _plugins = new TextBox();
    private readonly TextBox _timingCachePath = new TextBox();
    private readonly ComboBox _profilingVerbosity = new ComboBox();
    private readonly TextBox _builderOptimizationLevel = new TextBox();
    private readonly TextBox _maxAuxStreams = new TextBox();
    private readonly TextBox _deviceOrdinal = new TextBox();
    private readonly TextBox _dlaCore = new TextBox();
    private readonly TextBox _tacticSources = new TextBox();
    private readonly TextBox _maxNbTactics = new TextBox();
    private readonly ComboBox _tilingOptimizationLevel = new ComboBox();
    private readonly TextBox _l2LimitForTiling = new TextBox();
    private readonly ComboBox _quantizationFlags = new ComboBox();
    private readonly TextBox _memoryPoolSizes = new TextBox();
    private readonly TextBox _inputIoFormats = new TextBox();
    private readonly TextBox _outputIoFormats = new TextBox();
    private readonly TextBox _calibrationCachePath = new TextBox();
    private readonly TextBox _sparsity = new TextBox();
    private readonly CheckBox _allowGpuFallback = new CheckBox();
    private readonly CheckBox _directIo = new CheckBox();
    private readonly CheckBox _stronglyTyped = new CheckBox();
    private readonly TextBox _minTiming = new TextBox();
    private readonly TextBox _avgTiming = new TextBox();
    private readonly ComboBox _precisionConstraints = new ComboBox();
    private readonly TextBox _layerPrecisions = new TextBox();
    private readonly TextBox _layerOutputTypes = new TextBox();
    private readonly CheckBox _dumpRefit = new CheckBox();
    private readonly CheckBox _allowWeightStreaming = new CheckBox();
    private readonly TextBox _markDebug = new TextBox();
    private readonly CheckBox _dumpDebugTensors = new CheckBox();
    private readonly CheckBox _versionCompatible = new CheckBox();
    private readonly CheckBox _excludeLeanRuntime = new CheckBox();
    private readonly CheckBox _stripWeights = new CheckBox();
    private readonly CheckBox _refit = new CheckBox();
    private readonly TextBox _weightStreamingBudget = new TextBox();
    private readonly TextBox _exportTimingCachePath = new TextBox();
    private readonly CheckBox _safe = new CheckBox();
    private readonly CheckBox _consistency = new CheckBox();
    private readonly CheckBox _builderCache = new CheckBox();
    private readonly CheckBox _noBuilderCache = new CheckBox();
    private readonly CheckBox _dumpLayerInfo = new CheckBox();
    private readonly CheckBox _dumpProfile = new CheckBox();
    private readonly CheckBox _separateProfileRun = new CheckBox();
    private readonly TextBox _layerInfoPath = new TextBox();
    private readonly TextBox _reportPath = new TextBox();
    private readonly TextBox _evidenceSidecarPath = new TextBox();
    private readonly CheckBox _buildOnly = new CheckBox();
    private readonly CheckBox _skipInference = new CheckBox();
    private readonly CheckBox _dryRun = new CheckBox();
    private readonly NumericUpDown _iterations = new NumericUpDown();
    private readonly NumericUpDown _warmUp = new NumericUpDown();
    private readonly NumericUpDown _duration = new NumericUpDown();
    private readonly NumericUpDown _streams = new NumericUpDown();
    private readonly CheckBox _useCudaGraph = new CheckBox();
    private readonly CheckBox _noDataTransfers = new CheckBox();
    private readonly CheckBox _useSpinWait = new CheckBox();
    private readonly CheckBox _threads = new CheckBox();
    private readonly TextBox _avgRuns = new TextBox();
    private readonly TextBox _percentile = new TextBox();
    private readonly TextBox _sleepTime = new TextBox();
    private readonly TextBox _idleTime = new TextBox();
    private readonly TextBox _infStreams = new TextBox();
    private readonly TextBox _loadInputs = new TextBox();
    private readonly CheckBox _dumpOutput = new CheckBox();
    private readonly TextBox _dumpRawBindingsPath = new TextBox();
    private readonly TextBox _exportOutputPath = new TextBox();
    private readonly TextBox _exportTimesPath = new TextBox();
    private readonly TextBox _exportProfilePath = new TextBox();
    private readonly TextBox _saveProfilePath = new TextBox();
    private readonly TextBox _commandPreview = new TextBox();
    private readonly TextBox _log = new TextBox();

    public MainForm()
    {
        Text = "TensorRtExec";
        Width = 1040;
        Height = 860;
        MinimumSize = new Size(900, 720);
        Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point);

        TableLayoutPanel root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 3,
            RowCount = 49,
            Padding = new Padding(12),
            AutoScroll = true
        };
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 130));
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        root.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 110));
        Controls.Add(root);

        AddPathRow(root, 0, "ONNX", _onnxPath, OnBrowseOnnx);
        AddPathRow(root, 1, "Save Engine", _enginePath, OnBrowseSaveEngine);
        AddPathRow(root, 2, "Load Engine", _loadEnginePath, OnBrowseLoadEngine);
        AddLabeled(root, 3, "TensorRT", _tensorRtLine);
        _tensorRtLine.DropDownStyle = ComboBoxStyle.DropDownList;
        _tensorRtLine.Items.AddRange(new object[] { "8", "10", "11" });
        _tensorRtLine.SelectedItem = "10";

        FlowLayoutPanel precisionPanel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        ConfigureCheck(_fp16, "FP16");
        ConfigureCheck(_int8, "INT8");
        ConfigureCheck(_bf16, "BF16");
        ConfigureCheck(_fp8, "FP8");
        ConfigureCheck(_best, "Best");
        ConfigureCheck(_tf32, "TF32");
        _tf32.Checked = true;
        precisionPanel.Controls.AddRange(new Control[] { _fp16, _int8, _bf16, _fp8, _best, _tf32 });
        AddLabeled(root, 4, "Precision", precisionPanel);

        _workspace.Minimum = 0;
        _workspace.Maximum = 65536;
        _workspace.Value = 64;
        AddLabeled(root, 5, "Workspace MiB", _workspace);
        AddLabeled(root, 6, "Min Shapes", _minShapes);
        AddLabeled(root, 7, "Opt Shapes", _optShapes);
        AddLabeled(root, 8, "Max Shapes", _maxShapes);
        AddLabeled(root, 9, "Plugins", _plugins);
        AddPathRow(root, 10, "Timing Cache", _timingCachePath, OnBrowseTimingCache);

        _profilingVerbosity.DropDownStyle = ComboBoxStyle.DropDownList;
        _profilingVerbosity.Items.AddRange(new object[] { "layer_names_only", "detailed", "none" });
        _profilingVerbosity.SelectedItem = "layer_names_only";
        AddLabeled(root, 11, "Profiling", _profilingVerbosity);
        AddLabeled(root, 12, "Opt Level", _builderOptimizationLevel);
        AddLabeled(root, 13, "Aux Streams", _maxAuxStreams);
        AddLabeled(root, 14, "Device", _deviceOrdinal);
        AddLabeled(root, 15, "DLA Core", _dlaCore);

        FlowLayoutPanel deploymentPanel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        ConfigureCheck(_allowGpuFallback, "GPU fallback");
        ConfigureCheck(_directIo, "Direct IO");
        ConfigureCheck(_stronglyTyped, "Strongly typed");
        _tilingOptimizationLevel.DropDownStyle = ComboBoxStyle.DropDownList;
        _tilingOptimizationLevel.Items.AddRange(new object[] { "", "none", "fast", "moderate", "full" });
        _tilingOptimizationLevel.SelectedItem = "";
        _quantizationFlags.DropDownStyle = ComboBoxStyle.DropDownList;
        _quantizationFlags.Items.AddRange(new object[] { "", "none", "calibrateBeforeFusion" });
        _quantizationFlags.SelectedItem = "";
        deploymentPanel.Controls.AddRange(new Control[] { _allowGpuFallback, _directIo, _stronglyTyped });
        AddInlineField(deploymentPanel, "Max tactics", _maxNbTactics, 76);
        AddInlineField(deploymentPanel, "Tiling", _tilingOptimizationLevel, 92);
        AddInlineField(deploymentPanel, "L2 bytes", _l2LimitForTiling, 92);
        AddInlineField(deploymentPanel, "Quant", _quantizationFlags, 145);
        AddLabeled(root, 16, "Deployment", deploymentPanel);

        AddLabeled(root, 17, "Tactics", _tacticSources);
        AddLabeled(root, 18, "Mem Pools", _memoryPoolSizes);
        AddLabeled(root, 19, "Input IO", _inputIoFormats);
        AddLabeled(root, 20, "Output IO", _outputIoFormats);
        AddPathRow(root, 21, "Calib Cache", _calibrationCachePath, OnBrowseCalibrationCache);
        AddLabeled(root, 22, "Sparsity", _sparsity);
        AddLabeled(root, 23, "Min/Avg Timing", CreateTextPair(_minTiming, _avgTiming));
        _precisionConstraints.DropDownStyle = ComboBoxStyle.DropDownList;
        _precisionConstraints.Items.AddRange(new object[] { "", "none", "prefer", "obey" });
        _precisionConstraints.SelectedItem = "";
        AddLabeled(root, 24, "Precision Policy", _precisionConstraints);
        AddLabeled(root, 25, "Layer Precision", _layerPrecisions);
        AddLabeled(root, 26, "Layer Output Types", _layerOutputTypes);

        FlowLayoutPanel packagingPanel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        ConfigureCheck(_versionCompatible, "Version compatible");
        ConfigureCheck(_excludeLeanRuntime, "Exclude lean");
        ConfigureCheck(_stripWeights, "Strip weights");
        ConfigureCheck(_refit, "Refit");
        ConfigureCheck(_dumpRefit, "Dump refit");
        ConfigureCheck(_allowWeightStreaming, "Weight streaming");
        ConfigureCheck(_dumpDebugTensors, "Debug tensors");
        ConfigureCheck(_safe, "Safe");
        ConfigureCheck(_consistency, "Consistency");
        ConfigureCheck(_builderCache, "Builder cache");
        ConfigureCheck(_noBuilderCache, "No builder cache");
        packagingPanel.Controls.AddRange(new Control[] { _versionCompatible, _excludeLeanRuntime, _stripWeights, _refit, _dumpRefit, _allowWeightStreaming, _dumpDebugTensors, _safe, _consistency, _builderCache, _noBuilderCache });
        AddLabeled(root, 27, "Packaging", packagingPanel);
        AddLabeled(root, 28, "Weight Budget", _weightStreamingBudget);
        AddPathRow(root, 29, "Refit ONNX", _refitFromOnnxPath, OnBrowseRefitOnnx);
        AddPathRow(root, 30, "Refitted Engine", _saveRefittedEnginePath, OnBrowseSaveRefittedEngine);
        AddPathRow(root, 31, "Timing Export", _exportTimingCachePath, OnBrowseExportTimingCache);
        AddLabeled(root, 32, "Mark Debug", _markDebug);
        AddPathRow(root, 33, "Layer Info", _layerInfoPath, OnBrowseLayerInfo);
        AddPathRow(root, 34, "Report", _reportPath, OnBrowseReport);
        AddPathRow(root, 35, "Evidence", _evidenceSidecarPath, OnBrowseEvidenceSidecar);

        ConfigureNumeric(_iterations, 1, 100000, 10);
        ConfigureNumeric(_warmUp, 0, 3600000, 200);
        ConfigureNumeric(_duration, 0, 86400, 3);
        ConfigureNumeric(_streams, 1, 1024, 1);
        FlowLayoutPanel timingPanel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        timingPanel.Controls.AddRange(new Control[] { _iterations, _warmUp, _duration, _streams });
        AddLabeled(root, 36, "Runs/Warm/Duration/Streams", timingPanel);

        FlowLayoutPanel runtimeSwitchPanel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        ConfigureCheck(_useCudaGraph, "CUDA graph");
        ConfigureCheck(_noDataTransfers, "No transfers");
        ConfigureCheck(_useSpinWait, "Spin wait");
        ConfigureCheck(_threads, "Threads");
        ConfigureCheck(_dumpOutput, "Dump output");
        ConfigureCheck(_dumpLayerInfo, "Dump layer info");
        ConfigureCheck(_dumpProfile, "Dump profile");
        ConfigureCheck(_separateProfileRun, "Separate profile");
        runtimeSwitchPanel.Controls.AddRange(new Control[] { _useCudaGraph, _noDataTransfers, _useSpinWait, _threads, _dumpOutput, _dumpLayerInfo, _dumpProfile, _separateProfileRun });
        AddLabeled(root, 37, "Runtime Flags", runtimeSwitchPanel);

        AddLabeled(root, 38, "Inf Streams", _infStreams);
        AddLabeled(root, 39, "Avg/Percentile", CreateTextPair(_avgRuns, _percentile));
        AddLabeled(root, 40, "Sleep/Idle ms", CreateTextPair(_sleepTime, _idleTime));
        AddLabeled(root, 41, "Load Inputs", _loadInputs);
        AddLabeled(root, 42, "Raw Bindings", _dumpRawBindingsPath);
        AddLabeled(root, 43, "Output JSON", _exportOutputPath);
        AddLabeled(root, 44, "Times/Profile", CreateTextPair(_exportTimesPath, _exportProfilePath));
        AddLabeled(root, 45, "Save Profile", _saveProfilePath);

        FlowLayoutPanel modePanel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        ConfigureCheck(_buildOnly, "Build only");
        ConfigureCheck(_skipInference, "Skip inference");
        ConfigureCheck(_dryRun, "Dry run");
        _buildOnly.Checked = true;
        _skipInference.Checked = true;
        modePanel.Controls.AddRange(new Control[] { _buildOnly, _skipInference, _dryRun });
        AddLabeled(root, 46, "Mode", modePanel);

        Button previewButton = new Button { Text = "Preview", Dock = DockStyle.Fill, Height = 32 };
        previewButton.Click += OnPreview;
        root.Controls.Add(previewButton, 2, 46);

        _commandPreview.Dock = DockStyle.Fill;
        _commandPreview.ReadOnly = true;
        _commandPreview.WordWrap = false;
        root.Controls.Add(_commandPreview, 0, 47);
        root.SetColumnSpan(_commandPreview, 2);

        Button runButton = new Button { Text = "Run", Dock = DockStyle.Fill, Height = 32 };
        runButton.Click += OnRun;
        root.Controls.Add(runButton, 2, 47);

        _log.Dock = DockStyle.Fill;
        _log.Multiline = true;
        _log.ScrollBars = ScrollBars.Both;
        _log.ReadOnly = true;
        _log.WordWrap = false;
        root.Controls.Add(_log, 0, 48);
        root.SetColumnSpan(_log, 3);

        _minShapes.PlaceholderText = "input:1x3x640x640";
        _optShapes.PlaceholderText = "input:1x3x640x640";
        _maxShapes.PlaceholderText = "input:4x3x640x640";
        _plugins.PlaceholderText = "pluginA.dll;pluginB.dll";
        _builderOptimizationLevel.Text = "3";
        _maxAuxStreams.PlaceholderText = "0";
        _deviceOrdinal.PlaceholderText = "0";
        _dlaCore.PlaceholderText = "0";
        _tacticSources.PlaceholderText = "+CUBLAS,-CUDNN";
        _maxNbTactics.PlaceholderText = "0";
        _l2LimitForTiling.PlaceholderText = "256MiB";
        _memoryPoolSizes.PlaceholderText = "workspace:512,tacticDram:1024";
        _inputIoFormats.PlaceholderText = "fp16:chw";
        _outputIoFormats.PlaceholderText = "fp32:chw";
        _calibrationCachePath.PlaceholderText = "model.calib";
        _sparsity.PlaceholderText = "disable|enable|force";
        _minTiming.PlaceholderText = "1";
        _avgTiming.PlaceholderText = "8";
        _layerPrecisions.PlaceholderText = "conv1:fp16,head:fp32";
        _layerOutputTypes.PlaceholderText = "head:fp32";
        _weightStreamingBudget.PlaceholderText = "-2 | -1 | 50% | 512MiB";
        _saveRefittedEnginePath.PlaceholderText = "model.refitted.plan";
        _markDebug.PlaceholderText = "tensorA,tensorB";
        _exportTimingCachePath.PlaceholderText = "timing.cache";
        _evidenceSidecarPath.PlaceholderText = "model-evidence.sidecar.json";
        _infStreams.PlaceholderText = "1";
        _avgRuns.PlaceholderText = "10";
        _percentile.PlaceholderText = "99";
        _sleepTime.PlaceholderText = "0";
        _idleTime.PlaceholderText = "0";
        _loadInputs.PlaceholderText = "input:input.bin";
        _dumpRawBindingsPath.PlaceholderText = "bindings.raw";
        _exportOutputPath.PlaceholderText = "output.json";
        _exportTimesPath.PlaceholderText = "times.json";
        _exportProfilePath.PlaceholderText = "profile.json";
        _saveProfilePath.PlaceholderText = "profile.txt";
    }

    private static void ConfigureCheck(CheckBox checkBox, string text)
    {
        checkBox.Text = text;
        checkBox.AutoSize = true;
        checkBox.Margin = new Padding(0, 6, 18, 0);
    }

    private static void ConfigureNumeric(NumericUpDown input, int minimum, int maximum, int value)
    {
        input.Minimum = minimum;
        input.Maximum = maximum;
        input.Value = value;
        input.Width = 84;
    }

    private static FlowLayoutPanel CreateTextPair(TextBox first, TextBox second)
    {
        first.Width = 180;
        second.Width = 180;
        FlowLayoutPanel panel = new FlowLayoutPanel { Dock = DockStyle.Fill };
        panel.Controls.Add(first);
        panel.Controls.Add(second);
        return panel;
    }

    private static void AddInlineField(FlowLayoutPanel panel, string label, Control control, int width)
    {
        panel.Controls.Add(new Label { Text = label, AutoSize = true, Margin = new Padding(0, 8, 4, 0) });
        control.Width = width;
        control.Margin = new Padding(0, 3, 12, 0);
        panel.Controls.Add(control);
    }

    private static void AddPathRow(TableLayoutPanel root, int row, string label, TextBox textBox, EventHandler browseHandler)
    {
        AddLabeled(root, row, label, textBox);
        Button button = new Button { Text = "Browse", Dock = DockStyle.Fill };
        button.Click += browseHandler;
        root.Controls.Add(button, 2, row);
    }

    private static void AddLabeled(TableLayoutPanel root, int row, string label, Control control)
    {
        Label labelControl = new Label
        {
            Text = label,
            AutoSize = true,
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft
        };
        control.Dock = DockStyle.Fill;
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
        root.Controls.Add(labelControl, 0, row);
        root.Controls.Add(control, 1, row);
    }

    private void OnBrowseOnnx(object? sender, EventArgs e)
    {
        using OpenFileDialog dialog = new OpenFileDialog { Filter = "ONNX models (*.onnx)|*.onnx|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _onnxPath.Text = dialog.FileName;
        }
    }

    private void OnBrowseRefitOnnx(object? sender, EventArgs e)
    {
        using OpenFileDialog dialog = new OpenFileDialog { Filter = "ONNX models (*.onnx)|*.onnx|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _refitFromOnnxPath.Text = dialog.FileName;
        }
    }

    private void OnBrowseSaveRefittedEngine(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "TensorRT engines (*.plan;*.engine)|*.plan;*.engine|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _saveRefittedEnginePath.Text = dialog.FileName;
        }
    }

    private void OnBrowseSaveEngine(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "TensorRT engines (*.plan;*.engine)|*.plan;*.engine|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _enginePath.Text = dialog.FileName;
        }
    }

    private void OnBrowseLoadEngine(object? sender, EventArgs e)
    {
        using OpenFileDialog dialog = new OpenFileDialog { Filter = "TensorRT engines (*.plan;*.engine)|*.plan;*.engine|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _loadEnginePath.Text = dialog.FileName;
        }
    }

    private void OnBrowseTimingCache(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "Timing cache (*.cache)|*.cache|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _timingCachePath.Text = dialog.FileName;
        }
    }

    private void OnBrowseLayerInfo(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "Layer info (*.json)|*.json|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _layerInfoPath.Text = dialog.FileName;
        }
    }

    private void OnBrowseCalibrationCache(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "Calibration cache (*.calib;*.cache)|*.calib;*.cache|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _calibrationCachePath.Text = dialog.FileName;
        }
    }

    private void OnBrowseExportTimingCache(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "Timing cache (*.cache)|*.cache|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _exportTimingCachePath.Text = dialog.FileName;
        }
    }

    private void OnBrowseReport(object? sender, EventArgs e)
    {
        using SaveFileDialog dialog = new SaveFileDialog { Filter = "Reports (*.json;*.md)|*.json;*.md|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _reportPath.Text = dialog.FileName;
        }
    }

    private void OnBrowseEvidenceSidecar(object? sender, EventArgs e)
    {
        using OpenFileDialog dialog = new OpenFileDialog { Filter = "Evidence sidecars (*.json)|*.json|All files (*.*)|*.*" };
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            _evidenceSidecarPath.Text = dialog.FileName;
        }
    }

    private void OnRun(object? sender, EventArgs e)
    {
        try
        {
            TensorRtExecOptions options = CreateOptionsFromControls();
            _commandPreview.Text = options.ToArgumentLine();
            TensorRtExecReport report = new TensorRtExecService().Execute(options);
            _log.Text = FormatReportLog(report);
        }
        catch (Exception exception)
        {
            _log.Text = exception.Message;
        }
    }

    private static string FormatReportLog(TensorRtExecReport report)
    {
        List<string> lines = new List<string>();
        lines.AddRange(report.LogLines);

        if (!string.IsNullOrWhiteSpace(report.ReportPath))
        {
            lines.Add("TensorRtExec ReportPath=" + report.ReportPath);
        }

        if (!string.IsNullOrWhiteSpace(report.ProofClassification))
        {
            lines.Add("TensorRtExec ProofClassification=" + report.ProofClassification + " BuildEvidenceOnly=" + report.BuildEvidenceOnly.ToString(CultureInfo.InvariantCulture) + " DryRun=" + report.DryRun.ToString(CultureInfo.InvariantCulture));
        }

        if (!string.IsNullOrWhiteSpace(report.NormalizedCommandSha256))
        {
            lines.Add("TensorRtExec NormalizedCommandSha256=" + report.NormalizedCommandSha256);
        }

        if (!string.IsNullOrWhiteSpace(report.LoadEngineDiagnosticsState))
        {
            lines.Add("TensorRtExec LoadEngineDiagnosticsState=" + report.LoadEngineDiagnosticsState + " Attempted=" + report.LoadEngineDiagnosticsAttempted.ToString(CultureInfo.InvariantCulture) + " Succeeded=" + report.LoadEngineDiagnosticsSucceeded.ToString(CultureInfo.InvariantCulture));
            lines.Add("TensorRtExec LoadEngineDiagnosticsBoundary=" + report.LoadEngineDiagnosticsBoundary);
        }

        lines.Add("TensorRtExec WorkspaceBytes=" + report.WorkspaceBytes.ToString(CultureInfo.InvariantCulture));
        lines.Add("TensorRtExec BuilderConfigDeploymentSnapshot=" + report.BuilderConfigDeploymentSnapshotState + " Diagnostics=" + report.BuilderConfigDeploymentDiagnosticCount.ToString(CultureInfo.InvariantCulture));
        lines.Add("TensorRtExec ParserPreflightSnapshot=" + report.ParserPreflightSnapshotState + " Diagnostics=" + report.ParserPreflightDiagnosticCount.ToString(CultureInfo.InvariantCulture));
        lines.Add("TensorRtExec RefitPersistence=" + report.RefitPersistenceState + " Attempted=" + report.RefitPersistenceAttempted.ToString(CultureInfo.InvariantCulture) + " Succeeded=" + report.RefitPersistenceSucceeded.ToString(CultureInfo.InvariantCulture) + " Plan=" + report.PersistedRefittedEnginePath);
        lines.Add(report.Summary);
        return string.Join(Environment.NewLine, lines);
    }

    private void OnPreview(object? sender, EventArgs e)
    {
        try
        {
            _commandPreview.Text = CreateOptionsFromControls().ToArgumentLine();
        }
        catch (Exception exception)
        {
            _commandPreview.Text = exception.Message;
        }
    }

    private TensorRtExecOptions CreateOptionsFromControls()
    {
        return new TensorRtExecOptions(
            _onnxPath.Text,
            _enginePath.Text,
            _loadEnginePath.Text,
            _tensorRtLine.SelectedItem?.ToString() ?? "10",
            _fp16.Checked,
            _int8.Checked,
            _bf16.Checked,
            _tf32.Checked,
            (int)_workspace.Value,
            _buildOnly.Checked,
            _skipInference.Checked,
            _dryRun.Checked,
            (int)_iterations.Value,
            (int)_warmUp.Value,
            (int)_duration.Value,
            (int)_streams.Value,
            _useCudaGraph.Checked,
            _noDataTransfers.Checked,
            _useSpinWait.Checked,
            _threads.Checked ? 1 : (int?)null,
            ParseOptionalInt(_avgRuns.Text),
            ParseOptionalFloat(_percentile.Text),
            ParseOptionalInt(_sleepTime.Text),
            ParseOptionalInt(_idleTime.Text),
            ParseOptionalInt(_infStreams.Text),
            _loadInputs.Text,
            _dumpOutput.Checked,
            _dumpRawBindingsPath.Text,
            _exportOutputPath.Text,
            _exportTimesPath.Text,
            _exportProfilePath.Text,
            _saveProfilePath.Text,
            _minShapes.Text,
            _optShapes.Text,
            _maxShapes.Text,
            ParsePlugins(_plugins.Text),
            _timingCachePath.Text,
            _profilingVerbosity.SelectedItem?.ToString() ?? "layer_names_only",
            ParseRequiredInt(_builderOptimizationLevel.Text, 3),
            ParseOptionalInt(_maxAuxStreams.Text),
            ParseOptionalInt(_deviceOrdinal.Text),
            ParseOptionalInt(_dlaCore.Text),
            _allowGpuFallback.Checked,
            _tacticSources.Text,
            _memoryPoolSizes.Text,
            _inputIoFormats.Text,
            _outputIoFormats.Text,
            _calibrationCachePath.Text,
            _directIo.Checked,
            _sparsity.Text,
            _stronglyTyped.Checked,
            ParseOptionalInt(_minTiming.Text),
            ParseOptionalInt(_avgTiming.Text),
            _precisionConstraints.SelectedItem?.ToString() ?? string.Empty,
            _layerPrecisions.Text,
            _layerOutputTypes.Text,
            _fp8.Checked,
            _best.Checked,
            _dumpRefit.Checked,
            _allowWeightStreaming.Checked,
            _markDebug.Text,
            _dumpDebugTensors.Checked,
            _versionCompatible.Checked,
            _excludeLeanRuntime.Checked,
            _stripWeights.Checked,
            _refit.Checked,
            null,
            _exportTimingCachePath.Text,
            _safe.Checked,
            _consistency.Checked,
            _builderCache.Checked,
            _noBuilderCache.Checked,
            _dumpLayerInfo.Checked,
            _dumpProfile.Checked,
            _separateProfileRun.Checked,
            _layerInfoPath.Text,
            _reportPath.Text,
            _evidenceSidecarPath.Text,
            ParseOptionalInt(_maxNbTactics.Text),
            _tilingOptimizationLevel.SelectedItem?.ToString() ?? string.Empty,
            ParseOptionalMemoryBytes(_l2LimitForTiling.Text),
            _quantizationFlags.SelectedItem?.ToString() ?? string.Empty,
            _weightStreamingBudget.Text,
            _refitFromOnnxPath.Text,
            _saveRefittedEnginePath.Text);
    }

    private static string[] ParsePlugins(string value)
    {
        return string.IsNullOrWhiteSpace(value)
            ? Array.Empty<string>()
            : value.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries);
    }

    private static int ParseRequiredInt(string value, int defaultValue)
    {
        return string.IsNullOrWhiteSpace(value) ? defaultValue : int.Parse(value, CultureInfo.InvariantCulture);
    }

    private static int? ParseOptionalInt(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : int.Parse(value, CultureInfo.InvariantCulture);
    }

    private static float? ParseOptionalFloat(string value)
    {
        return string.IsNullOrWhiteSpace(value) ? null : float.Parse(value, CultureInfo.InvariantCulture);
    }

    private static ulong? ParseOptionalMemoryBytes(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string trimmed = value.Trim();
        string numberText = trimmed;
        decimal multiplier = 1024m * 1024m;
        if (TryTrimSuffix(trimmed, "gib", out numberText) || TryTrimSuffix(trimmed, "gb", out numberText))
        {
            multiplier = 1024m * 1024m * 1024m;
        }
        else if (TryTrimSuffix(trimmed, "mib", out numberText) || TryTrimSuffix(trimmed, "mb", out numberText))
        {
            multiplier = 1024m * 1024m;
        }
        else if (TryTrimSuffix(trimmed, "kib", out numberText) || TryTrimSuffix(trimmed, "kb", out numberText))
        {
            multiplier = 1024m;
        }
        else if (TryTrimSuffix(trimmed, "b", out numberText))
        {
            multiplier = 1m;
        }

        decimal parsed = decimal.Parse(numberText.Trim(), CultureInfo.InvariantCulture);
        return checked((ulong)(parsed * multiplier));
    }

    private static bool TryTrimSuffix(string value, string suffix, out string withoutSuffix)
    {
        if (value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
        {
            withoutSuffix = value.Substring(0, value.Length - suffix.Length);
            return true;
        }

        withoutSuffix = value;
        return false;
    }
}
