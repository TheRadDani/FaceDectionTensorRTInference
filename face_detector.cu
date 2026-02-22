#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <memory>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class FaceDetector {
private:
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    Logger logger;
    std::vector<void*> buffers;
    std::vector<size_t> buffer_sizes;
    cudaStream_t stream;
    int input_idx, output_idx;

public:
    FaceDetector(const std::string& engine_path) 
        : runtime(nullptr), engine(nullptr), context(nullptr), 
          input_idx(-1), output_idx(-1) {
        
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file.good()) {
            throw std::runtime_error("Cannot open engine file: " + engine_path);
        }
        
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(engine_size);
        engine_file.read(engine_data.data(), engine_size);
        engine_file.close();
        
        runtime = createInferRuntime(logger);
        if (!runtime) {
            throw std::runtime_error("Failed to create runtime");
        }
        
        engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
        if (!engine) {
            delete runtime;
            throw std::runtime_error("Failed to deserialize engine");
        }
        
        context = engine->createExecutionContext();
        if (!context) {
            delete engine;
            delete runtime;
            throw std::runtime_error("Failed to create execution context");
        }
        
        allocateBuffers();
        
        std::cout << "Engine loaded successfully" << std::endl;
    }
    
    ~FaceDetector() {
        for (auto buffer : buffers) {
            cudaFree(buffer);
        }
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
        cudaStreamDestroy(stream);
    }
    
    void allocateBuffers() {
        int32_t nb_io_tensors = engine->getNbIOTensors();
        buffers.resize(nb_io_tensors);
        buffer_sizes.resize(nb_io_tensors);
        
        std::cout << "Number of IO tensors: " << nb_io_tensors << std::endl;
        
        for (int32_t i = 0; i < nb_io_tensors; ++i) {
            const char* tensor_name = engine->getIOTensorName(i);
            Dims dims = engine->getTensorShape(tensor_name);
            DataType dtype = engine->getTensorDataType(tensor_name);
            TensorIOMode tensor_mode = engine->getTensorIOMode(tensor_name);
            
            std::cout << "Tensor " << i << ": " << tensor_name 
                     << " Mode: " << (tensor_mode == TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;
            
            int64_t total_size = 1;
            for (int32_t j = 0; j < dims.nbDims; ++j) {
                total_size *= dims.d[j];
                std::cout << "  Dim[" << j << "]: " << dims.d[j] << std::endl;
            }
            
            size_t buffer_size = total_size * getTypeSize(dtype);
            buffer_sizes[i] = buffer_size;
            
            cudaMalloc(&buffers[i], buffer_size);
            context->setTensorAddress(tensor_name, buffers[i]);
            
            if (tensor_mode == TensorIOMode::kINPUT) {
                input_idx = i;
            } else {
                output_idx = i;
            }
        }
    }
    
    static size_t getTypeSize(DataType type) {
        switch (type) {
            case DataType::kFLOAT: return 4;
            case DataType::kHALF: return 2;
            case DataType::kINT32: return 4;
            case DataType::kINT8: return 1;
            case DataType::kBOOL: return 1;
            case DataType::kUINT8: return 1;
            default: return 4;
        }
    }
    
    std::vector<Detection> detect(const cv::Mat& image) {
        // Preprocess image
        cv::Mat blob = preprocessImage(image);
        
        // Copy input to GPU
        size_t input_size = blob.total() * blob.elemSize();
        cudaMemcpyAsync(buffers[input_idx], blob.data, input_size, 
                       cudaMemcpyHostToDevice, stream);
        
        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        bool success = context->enqueueV3(stream);
        if (!success) {
            throw std::runtime_error("Failed to enqueue inference");
        }
        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        
        // Copy output from GPU
        std::vector<float> output_data(buffer_sizes[output_idx] / sizeof(float));
        std::cout << "Output buffer size: " << buffer_sizes[output_idx] << " bytes" << std::endl;
        std::cout << "Output elements: " << output_data.size() << std::endl;
        
        cudaMemcpyAsync(output_data.data(), buffers[output_idx], 
                       buffer_sizes[output_idx], cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Parse detections
        auto detections = parseDetections(output_data, image.rows, image.cols);
        
        // Apply NMS with more aggressive threshold
        auto nms_detections = nmsDetections(detections, 0.4f);
        std::cout << "Detections after NMS: " << nms_detections.size() << std::endl;
        
        return nms_detections;
    }
    
private:
    cv::Mat preprocessImage(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(640, 640));
        
        cv::Mat blob;
        cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(640, 640),
                              cv::Scalar(0, 0, 0), true, false, CV_32F);
        
        return blob;
    }
    
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    static float iouDetections(const Detection& a, const Detection& b) {
        float x1_max = std::max(a.x1, b.x1);
        float y1_max = std::max(a.y1, b.y1);
        float x2_min = std::min(a.x2, b.x2);
        float y2_min = std::min(a.y2, b.y2);
        
        if (x2_min < x1_max || y2_min < y1_max) {
            return 0.0f;
        }
        
        float intersection = (x2_min - x1_max) * (y2_min - y1_max);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        float union_area = area_a + area_b - intersection;
        
        return union_area > 0 ? intersection / union_area : 0.0f;
    }
    
    std::vector<Detection> nmsDetections(const std::vector<Detection>& detections, 
                                        float iou_threshold) {
        std::vector<Detection> result;
        if (detections.empty()) return result;
        
        // Sort detections by confidence
        std::vector<Detection> sorted = detections;
        std::sort(sorted.begin(), sorted.end(), 
                 [](const Detection& a, const Detection& b) {
                     return a.confidence > b.confidence;
                 });
        
        std::vector<bool> keep(sorted.size(), true);
        
        for (size_t i = 0; i < sorted.size(); ++i) {
            if (!keep[i]) continue;
            
            result.push_back(sorted[i]);
            
            for (size_t j = i + 1; j < sorted.size(); ++j) {
                if (keep[j]) {
                    float iou = iouDetections(sorted[i], sorted[j]);
                    if (iou > iou_threshold) {
                        keep[j] = false;
                    }
                }
            }
        }
        
        return result;
    }
    
    std::vector<Detection> parseDetections(const std::vector<float>& output, 
                                          int img_h, int img_w) {
        std::vector<Detection> detections;
        
        if (output.empty()) {
            std::cout << "Output is empty!" << std::endl;
            return detections;
        }
        
        std::cout << "\nParsing detections..." << std::endl;
        std::cout << "Image size: " << img_w << "x" << img_h << std::endl;
        
        // Output format: [1, 5, 8400] -> reshape to [5, 8400]
        // Channels: [x, y, w, h, conf] for each of 8400 positions
        // Coordinates are in 640x640 space (model input size)
        int num_positions = 8400;
        int num_channels = 5;
        
        // Verify output size
        if (output.size() != num_positions * num_channels) {
            std::cout << "Warning: expected " << (num_positions * num_channels) 
                     << " elements, got " << output.size() << std::endl;
            num_positions = output.size() / num_channels;
        }
        
        std::cout << "Output shape: [" << num_channels << ", " << num_positions << "]" << std::endl;
        
        float conf_threshold = 0.6f;  // Balanced confidence threshold
        float scale_x = img_w / 640.0f;    // Scale from 640x640 to image size
        float scale_y = img_h / 640.0f;
        
        for (int i = 0; i < num_positions; ++i) {
            // Extract channel values for this position
            float x = output[0 * num_positions + i];      // channel 0
            float y = output[1 * num_positions + i];      // channel 1
            float w = output[2 * num_positions + i];      // channel 2
            float h = output[3 * num_positions + i];      // channel 3
            float conf_logit = output[4 * num_positions + i];  // channel 4
            
            // Apply sigmoid to get confidence in [0, 1] range
            float conf = sigmoid(conf_logit);
            
            if (conf > conf_threshold && w > 5 && h > 5) {
                // Scale coordinates from 640x640 to actual image size
                float x_scaled = x * scale_x;
                float y_scaled = y * scale_y;
                float w_scaled = w * scale_x;
                float h_scaled = h * scale_y;
                
                Detection det;
                det.x1 = std::max(0.0f, x_scaled - w_scaled / 2.0f);
                det.y1 = std::max(0.0f, y_scaled - h_scaled / 2.0f);
                det.x2 = std::min((float)img_w, x_scaled + w_scaled / 2.0f);
                det.y2 = std::min((float)img_h, y_scaled + h_scaled / 2.0f);
                det.confidence = conf;
                det.class_id = 0;
                
                // Validate box dimensions
                if (det.x2 > det.x1 && det.y2 > det.y1) {
                    detections.push_back(det);
                }
            }
        }
        
        std::cout << "Detections before NMS: " << detections.size() << std::endl;
        
        return detections;
    }
};

int main(int argc, char* argv[]) {
    try {
        // Load image
        std::string image_path = "lenna.png";
        cv::Mat image = cv::imread(image_path);
        
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return -1;
        }
        
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        // Initialize detector
        FaceDetector detector("face-detector.engine");
        
        // Run detection
        auto detections = detector.detect(image);
        
        std::cout << "Detected " << detections.size() << " faces" << std::endl;
        
        // Draw bounding boxes
        for (const auto& det : detections) {
            cv::rectangle(image, cv::Point(det.x1, det.y1), 
                         cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
            std::string label = "Face: " + std::to_string(det.confidence).substr(0, 4);
            cv::putText(image, label,
                       cv::Point(det.x1, det.y1 - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
        
        // Save result
        cv::imwrite("result.png", image);
        std::cout << "Result saved to result.png" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}