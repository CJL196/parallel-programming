#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Helper for checking CUDA calls
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/**
 * @brief 2D Convolution Kernel (Direct Method)
 * 
 * @param input Pointer to the input data on the device.
 * @param kernel Pointer to the kernel data on the device.
 * @param output Pointer to the output data on the device.
 * @param input_h Height of the input matrix.
 * @param input_w Width of the input matrix.
 * @param kernel_h Height of the kernel.
 * @param kernel_w Width of the kernel.
 * @param channels Number of channels in the input and kernel.
 * @param stride The stride of the convolution.
 * @param padding The padding applied to the input.
 * @param output_h Height of the output matrix.
 * @param output_w Width of the output matrix.
 */
__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                               int input_h, int input_w,
                               int kernel_h, int kernel_w,
                               int channels, int stride, int padding,
                               int output_h, int output_w) {

    // Calculate the output pixel coordinates handled by this thread
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to ensure the thread is within the output matrix dimensions
    if (out_x < output_w && out_y < output_h) {
        float acc = 0.0f;
        
        // Calculate the top-left corner of the receptive field in the input
        int in_base_y = out_y * stride - padding;
        int in_base_x = out_x * stride - padding;

        // Perform the convolution: sum of element-wise products
        for (int c = 0; c < channels; ++c) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_y = in_base_y + kh;
                    int in_x = in_base_x + kw;

                    // Check if the current position is within the actual input dimensions (not in the padding)
                    if (in_y >= 0 && in_y < input_h && in_x >= 0 && in_x < input_w) {
                        int input_idx = (in_y * input_w + in_x) * channels + c;
                        int kernel_idx = (kh * kernel_w + kw) * channels + c;
                        acc += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        // Write the final accumulated value to the output matrix
        output[out_y * output_w + out_x] = acc;
    }
}

// Utility function to print the top-left corner of a matrix for verification
void print_matrix(const float* matrix, int height, int width, const std::string& name) {
    std::cout << name << " (top-left 5x5):" << std::endl;
    for (int i = 0; i < std::min(5, height); ++i) {
        for (int j = 0; j < std::min(5, width); ++j) {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void warmup_gpu() {
    // Some small, reasonable size for warmup
    const int warmup_size = 256;
    const int channels = 3;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;

    int output_h = (warmup_size - kernel_size + 2 * padding) / stride + 1;
    int output_w = (warmup_size - kernel_size + 2 * padding) / stride + 1;

    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, (size_t)warmup_size * warmup_size * channels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, (size_t)kernel_size * kernel_size * channels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, (size_t)output_h * output_w * sizeof(float)));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int i = 0; i < 5; ++i) {
        conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output,
                                                     warmup_size, warmup_size,
                                                     kernel_size, kernel_size,
                                                     channels, stride, padding,
                                                     output_h, output_w);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    std::cout << "GPU warmup complete." << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_size> <stride>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 512 1" << std::endl;
        return 1;
    }

    warmup_gpu();

    // --- Configuration ---
    int input_size = std::atoi(argv[1]);
    int stride = std::atoi(argv[2]);

    int input_h = input_size;
    int input_w = input_size;
    int channels = 3;
    int kernel_h = 3;
    int kernel_w = 3;
    // Padding=1 is a common choice for 3x3 kernels.
    int padding = 1; 

    // --- Calculate Output Dimensions ---
    int output_h = (input_h - kernel_h + 2 * padding) / stride + 1;
    int output_w = (input_w - kernel_w + 2 * padding) / stride + 1;

    if (output_h <= 0 || output_w <= 0) {
        std::cerr << "Error: Invalid dimensions for output matrix. Check input size, kernel size, padding and stride." << std::endl;
        return 1;
    }
    
    std::cout << "Input Size: " << input_h << "x" << input_w << ", Channels: " << channels << std::endl;
    std::cout << "Kernel Size: " << kernel_h << "x" << kernel_w << std::endl;
    std::cout << "Stride: " << stride << ", Padding: " << padding << std::endl;
    std::cout << "Output Size: " << output_h << "x" << output_w << std::endl;

    // --- Memory Allocation ---
    size_t input_bytes = (size_t)input_h * input_w * channels * sizeof(float);
    size_t kernel_bytes = (size_t)kernel_h * kernel_w * channels * sizeof(float);
    size_t output_bytes = (size_t)output_h * output_w * sizeof(float);

    float *h_input = new float[input_h * input_w * channels];
    float *h_kernel = new float[kernel_h * kernel_w * channels];
    float *h_output = new float[output_h * output_w];

    // --- Initialization ---
    srand(time(0));
    for (int i = 0; i < input_h * input_w * channels; ++i) h_input[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < kernel_h * kernel_w * channels; ++i) h_kernel[i] = (float)rand() / RAND_MAX;

    // --- CUDA Device Allocation and Data Transfer ---
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice));

    // --- Kernel Launch ---
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_w + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (output_h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    conv2d_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_kernel, d_output,
                                                 input_h, input_w,
                                                 kernel_h, kernel_w,
                                                 channels, stride, padding,
                                                 output_h, output_w);

    CHECK_CUDA(cudaEventRecord(stop));
    
    // Check for any errors during kernel launch
    CHECK_CUDA(cudaGetLastError());

    // --- Synchronize and Calculate Time ---
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // --- Copy Result Back to Host ---
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    
    // --- Verification ---
    print_matrix(h_output, output_h, output_w, "Output Matrix");

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    return 0;
}
