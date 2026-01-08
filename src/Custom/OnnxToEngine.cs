//using JYPPX.TensorRtSharp.Nvinfer;
//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.Linq;
//using System.Net;
//using System.Reflection;
//using System.Text;
//using static System.Runtime.InteropServices.JavaScript.JSType;

//namespace JYPPX.TensorRtSharp.Custom
//{

//    public class ShapeProfileManager
//    {
//        /// <summary>
//        /// 形状配置文件
//        /// </summary>
//        public struct ShapeProfile
//        {
//            public string Name;
//            public List<int> Shape;
//            public TrtOptProfileSelector TrtOptProfile;
//            // 提供一个便捷的只读属性，防止返回 null
//            public IReadOnlyList<int> ReadOnlyShape => Shape ?? (IReadOnlyList<int>)Array.Empty<int>();
//        }
//        /// <summary>
//        /// 用于字典的复合键，确保每个 (Name, TrtOptProfile) 组合是唯一的
//        /// </summary>
//        private readonly struct ProfileKey : IEquatable<ProfileKey>
//        {
//            public readonly string Name;
//            public readonly TrtOptProfileSelector Selector;
//            public ProfileKey(string name, TrtOptProfileSelector selector)
//            {
//                Name = name;
//                Selector = selector;
//            }
//            // 重写 Equals 和 GetHashCode 以确保字典可以正确地比较键
//            public override bool Equals(object obj) => obj is ProfileKey other && this.Equals(other);
//            public bool Equals(ProfileKey other)
//            {
//                // 使用 string.Equals 进行比较，并处理 Name 为 null 的情况
//                return string.Equals(this.Name, other.Name, StringComparison.Ordinal) && this.Selector == other.Selector;
//            }

//        }
//        // --- 核心改动：使用 Dictionary 替代 List ---
//        /// <summary>
//        /// 使用字典来存储形状配置，以实现高效的查找和更新。
//        /// Key: 由 (Name, Selector) 组成的复合键
//        /// Value: 对应的 ShapeProfile
//        /// </summary>
//        private readonly Dictionary<ProfileKey, ShapeProfile> _shapeProfileMap = new Dictionary<ProfileKey, ShapeProfile>();
//        /// <summary>
//        /// (可选) 如果仍然需要列表形式的访问，可以提供这个属性
//        /// 它会从字典中实时生成列表，因此性能不如直接使用字典。
//        /// </summary>
//        public List<ShapeProfile> ShapeProfiles => _shapeProfileMap.Values.ToList();
//        /// <summary>
//        /// 设置或更新输入张量的形状范围。
//        /// 如果指定的配置已存在，则覆盖它；否则，添加新的配置。
//        /// </summary>
//        /// <param name="trtOptProfile">优化配置选择器</param>
//        /// <param name="name">输入张量的名称</param>
//        /// <param name="shape">要设置的形状</param>
//        public void SetShapeRange(TrtOptProfileSelector trtOptProfile, string name, List<int> shape)
//        {
//            var key = new ProfileKey(name, trtOptProfile);
//            var profile = new ShapeProfile
//            {
//                Name = name,
//                Shape = shape,
//                TrtOptProfile = trtOptProfile
//            };
//            // 字典的索引器设置操作会自动处理“添加”和“更新”
//            // 如果 key 不存在，它会添加一个新的键值对
//            // 如果 key 已存在，它会用新的 value 覆盖旧的 value
//            _shapeProfileMap[key] = profile;
//        }
//        /// <summary>
//        /// 尝试获取指定张量和选择器的形状配置。
//        /// </summary>
//        /// <param name="trtOptProfile">优化配置选择器</param>
//        /// <param name="name">输入张量的名称</param>
//        /// <returns>如果找到，返回对应的 ShapeProfile；否则返回 null</returns>
//        public ShapeProfile? GetShapeRange(TrtOptProfileSelector trtOptProfile, string name)
//        {
//            var key = new ProfileKey(name, trtOptProfile);
//            if (_shapeProfileMap.TryGetValue(key, out var profile))
//            {
//                return profile;
//            }
//            return null;
//        }

//        public int Size() => _shapeProfileMap.Count;
//        /// <summary>
//        /// 清除所有形状配置。
//        /// </summary>
//        public void Clear()
//        {
//            _shapeProfileMap.Clear();
//        }
//    }

//    public class OnnxToEngine
//    {

     
//        public ShapeProfileManager ShapeProfiles { get; set; } = new ShapeProfileManager();

//        public int DeviceId { get; set; } = 0;

//        public string ModelPath { get; set; }

//        // builder
//        bool stronglyTyped  =  false;
//        bool pluginInstanceNorm = false;
//        bool enableUInt8AsymmetricQuantizationDLA = false;

//        public OnnxToEngine()
//        {
//        }

//        public void Export() 
//        {
//            DeviceSettings.SetCudaDevice(DeviceId);
//            Logger.Instance.INFO("");
//            Logger.Instance.INFO("TensorRT version: " + TrtVersion.Version);

//            Builder builder = new Builder();




//            Logger.Instance.INFO("Start parsing network model.");
//            Stopwatch tBegin = new Stopwatch();
//            tBegin.Start();


//            TrtNetworkDefinitionCreationFlag networkFlags = (TrtNetworkDefinitionCreationFlag)((stronglyTyped) ? 1U  << (int)TrtNetworkDefinitionCreationFlag.kSTRONGLY_TYPED : 0U);
//            NetworkDefinition network = builder.createNetworkV2(networkFlags);

            
//            OnnxParser parser = new OnnxParser(network);

//            // kNATIVE_INSTANCENORM is ON by default in the parser and must be cleared to use the plugin implementation.
//            if (pluginInstanceNorm)
//            {
//                parser.clearFlag(TrtOnnxParserFlag.kNATIVE_INSTANCENORM);
//            }
//            if (enableUInt8AsymmetricQuantizationDLA)
//            {
//                parser.setFlag(TrtOnnxParserFlag.kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA);
//            }
//            if (!parser.parseFromFile(ModelPath, 2))
//            {
//                Logger.Instance.ERROR( "Failed to parse onnx file" );
//                    parser.Dispose();
//                parser = new OnnxParser(network);
//            }
//            tBegin.Stop();

//            Logger.Instance.INFO("Finished parsing network model. Parse time: " + tBegin.ElapsedMilliseconds + " ms");


//            BuilderConfig config = builder.createBuilderConfig();
            
//            List<List<byte>> sparseWeights = new List<List<byte>>();

//            //SMP_RETVAL_IF_FALSE(
//            //    setupNetworkAndConfig(build, sys, builder, *env.network, *config, calibrator, err, sparseWeights),
//            //    "Network And Config setup failed", false, err);

//            List<OptimizationProfile> profiles = new List<OptimizationProfile>();

//             for (int i = 0; i < ShapeProfiles.Size(); ++i)
//            {
//                profiles.Add(builder.createOptimizationProfile());
//            }



//            bool hasDynamicShapes= false
//            ;

//            bool broadcastInputFormats = broadcastIOFormats(build.inputFormats, network.getNbInputs());

//            bool broadcast = formats.size() == 1;
//            bool validFormatsCount = broadcast || (formats.size() == nbBindings);
//            if (!formats.empty() && !validFormatsCount)
//            {
//                if (isInput)
//                {
//                    throw std::invalid_argument(
//                        "The number of inputIOFormats must match network's inputs or be one for broadcasting.");
//                }

//                throw std::invalid_argument(
//                    "The number of outputIOFormats must match network's outputs or be one for broadcasting.");
//            }
//            return broadcast;




//            std::unique_ptr<ITimingCache> timingCache{ }
//            ;
//            // Try to load cache from file. Create a fresh cache if the file doesn't exist
//            if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
//            {
//                timingCache = samplesCommon::buildTimingCacheFromFile(gLogger.getTRTLogger(), *config, build.timingCacheFile);
//            }

//            // CUDA stream used for profiling by the builder.
//            auto profileStream = samplesCommon::makeCudaStream();
//            SMP_RETVAL_IF_FALSE(profileStream != nullptr, "Cuda stream creation failed", false, err);
//            config->setProfileStream(*profileStream);

//            auto const tBegin = std::chrono::high_resolution_clock::now();

//            if (!(build.safe || build.buildDLAStandalone) && build.save)
//            {
//                auto const engineFile = build.engine;
//                FileStreamWriter writer(engineFile);
//                SMP_RETVAL_IF_FALSE(builder.buildSerializedNetworkToStream(*env.network, *config, writer),
//                    "Engine could not be created from network", false, err);
//                auto const engineSize = writer.finalize();
//                std::vector<uint8_t> streamEngine(engineSize, 0);
//                std::ifstream reader(engineFile, std::ios::binary);
//                SMP_RETVAL_IF_FALSE((reader.is_open() && reader.good()), "Failed to open engine file for reading", false, err);
//                reader.read(reinterpret_cast<char*>(streamEngine.data()), engineSize);
//                SMP_RETVAL_IF_FALSE((!reader.fail()), "Error when reading engine file", false, err);
//                reader.close();
//                sample::gLogInfo << "Created engine with size: " << (engineSize / 1.0_MiB) << " MiB" << std::endl;
//                env.engine.setBlob(std::move(streamEngine));
//            }
//            else
//            {
//                std::unique_ptr<IHostMemory> serializedEngine{ builder.buildSerializedNetwork(*env.network, *config)}
//                ;
//                SMP_RETVAL_IF_FALSE(serializedEngine != nullptr, "Engine could not be created from network", false, err);
//                sample::gLogInfo << "Created engine with size: " << (serializedEngine->size() / 1.0_MiB) << " MiB" << std::endl;
//                if (build.safe && build.consistency)
//                {
//                    if (!checkSafeEngine(serializedEngine->data(), serializedEngine->size()))
//                    {
//                        sample::gLogError << "Consistency validation is not supported." << std::endl;
//                        return false;
//                    }
//                }
//                env.engine.setBlob(serializedEngine);
//            }

//            auto const tEnd = std::chrono::high_resolution_clock::now();
//            float const buildTime = std::chrono::duration<float>(tEnd - tBegin).count();
//            sample::gLogInfo << "Engine built in " << buildTime << " sec." << std::endl;

//            if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
//            {
//                auto timingCache = config->getTimingCache();
//                samplesCommon::updateTimingCacheFile(gLogger.getTRTLogger(), build.timingCacheFile, timingCache, builder);
//            }

//            return true;

//        }
//    }
//}
