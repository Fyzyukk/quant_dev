#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "[ERROR] " << msg << std::endl;
        }
    }
};

// 读取 TensorRT engine 文件
vector<unsigned char> loadEngineFile(const string& enginePath) {
    ifstream file(enginePath, ios::binary);
    if (!file) {
        cerr << "Error: Could not open engine file: " << enginePath << endl;
        exit(EXIT_FAILURE);
    }
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<unsigned char> engineData(size);
    file.read(reinterpret_cast<char*>(engineData.data()), size);
    return engineData;
}

struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj; 
    }
};

int main(int argc, char** argv) {
    std::string enginePath = "./rrdbnet_sim_quant_conv.trt";  // 直接指定 engine 文件路径

    // 加载 engine
    std::vector<unsigned char> engineData = loadEngineFile(enginePath);
    
    // 1. 读取 engine 文件
    Logger logger;

    unique_ptr<IRuntime, TRTDeleter> runtime(createInferRuntime(logger));
    unique_ptr<ICudaEngine, TRTDeleter> engine(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    unique_ptr<IExecutionContext, TRTDeleter> context(engine->createExecutionContext());

    if (!context) {
        cerr << "Error: Failed to create execution context." << endl;
        return EXIT_FAILURE;
    }
    // auto runtime = unique_ptr<IRuntime>(createInferRuntime(logger));
    // auto engine = unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    // auto context = unique_ptr<IExecutionContext>(engine->createExecutionContext());

    // 2. 获取输入/输出 tensor 名称并分配 GPU 内存
    const char* inputTensorName = engine->getIOTensorName(0);
    const char* outputTensorName = engine->getIOTensorName(1);
    auto inputDims = context->getTensorShape(inputTensorName);
    auto outputDims = context->getTensorShape(outputTensorName);
    size_t tensorSize = 1 * 3 * 256 * 256 * sizeof(float); // 1x3x256x256

    vector<float> input_h(1 * 3 * 256 * 256);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < input_h.size(); ++i) {
        input_h[i] = dis(gen);
    }
    vector<float> output_h(1 * 3 * 256 * 256);

    float* input_d;
    float* output_d;
    cudaMalloc(&input_d, tensorSize);
    cudaMalloc(&output_d, tensorSize);

    // 3. 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(input_d, input_h.data(), tensorSize, cudaMemcpyHostToDevice, stream);

    // 4. 运行推理
    float* bindings[] = {input_d, output_d};
    if (!context->executeV2(reinterpret_cast<void**>(bindings))) {
        cerr << "Error: Inference failed." << endl;
        return EXIT_FAILURE;
    }

    cout << "Inference completed successfully." << endl;

    cudaMemcpyAsync(output_h.data(), output_d, tensorSize, cudaMemcpyDeviceToHost, stream);
    
    // 同步流，确保所有操作完成
    cudaStreamSynchronize(stream);

    // 5. 释放资源
    cudaFree(input_d);
    cudaFree(output_d);
    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}
