# Classification Local Package Consumer

This template is copied to a repository-external workspace by
`eng/Test-ClassificationLocalPackageConsumer.ps1`. It references only the managed API,
Classification extension, and runtime-key-selected bridge-only packages. It never uses a
`ProjectReference` or a direct assembly path.

The runner uses the official torchvision ResNet18 ONNX stored under the outer `models` directory,
a redistributable input image, and independently generated ONNX Runtime references. TensorRT and
CUDA remain user-installed dependencies. A successful run is local-package engineering evidence;
it is not public-feed, post-publish, redistribution-approval, or release evidence.
