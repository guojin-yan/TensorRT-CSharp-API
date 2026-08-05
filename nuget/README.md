# NuGet package assets

All new TensorRtSharp4.0 NuGet packages use the following canonical package assets:

- package icon: `nuget/logo.jpg`, packed as `/logo.jpg` and declared with `PackageIcon=logo.jpg`;
- package README: the repository root English `README.md`, packed as `/README.md` and declared with `PackageReadmeFile=README.md`.

Package-specific README files may remain as source-tree documentation, but they must not replace the root English README in a published package. These rules apply to the managed API, YoloVision, Classification, and every bridge package. Packaging and validation do not authorize publication.
