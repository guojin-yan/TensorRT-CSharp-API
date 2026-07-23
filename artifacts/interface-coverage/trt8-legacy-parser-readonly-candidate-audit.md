# TRT8 Legacy Parser Readonly Candidate Audit

The current readonly candidate plan contains six low-risk rows, all from TensorRT 8 legacy parsers. Both local
TensorRT 8.6.1.6 packages expose `createUffParser`, `createCaffeParser`, and both protobuf shutdown symbols in
`nvparsers.lib` and `nvparsers.dll`. The DLL depends only on `nvinfer.dll` and `KERNEL32.dll`.

The selected safe alternatives do not expose `IUffParser*`, `ICaffeParser*`, `IBinaryProtoBlob*`, or
`IBinaryProtoBlob::getData()` pointers. UFF parser lifetime remains within one native call and returns three version
scalars. Caffe parser/blob lifetime remains within each native call and copies dimensions, data type, and bytes into a
caller buffer. The bridge never calls process-global `shutdownProtobufLibrary`.

UFF parsing, input/output registration, plugin namespace, Caffe network parsing, plugin factories, error-recorder
mutation, protobuf buffer mutation, and process-global protobuf shutdown remain deferred.
