#include <cuda.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void resizeKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight, bool antialiasing) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX >= outputWidth || outY >= outputHeight) return;
    if (antialiasing) {
        float scaleWidth = (float)inputWidth / outputWidth;
        float scaleHeight = (float)inputHeight / outputHeight;

        int startX = (int)(scaleWidth * outX);
        int startY = (int)(scaleHeight * outY);
        int endX = (int)(scaleWidth * (outX + 1));
        int endY = (int)(scaleHeight * (outY + 1));

        float pixelValue = 0;
        int count = 0;

        for (int y = startY; y < endY; ++y) {
            for (int x = startX; x < endX; ++x) {
                pixelValue += input[y * inputWidth + x];
                count++;
            }
        }

        pixelValue /= count;
        output[outY * outputWidth + outX] = (unsigned char)pixelValue;
    }
    else {
        // Find the nearest neighbor in the input image
        int inX = outX * inputWidth / outputWidth;
        int inY = outY * inputHeight / outputHeight;

        int inIndex = inY * inputWidth + inX;
        int outIndex = outY * outputWidth + outX;

        // Assign the pixel value from the nearest neighbor
        output[outIndex] = input[inIndex];
    }
}

__global__ void upsampleKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX >= outputWidth || outY >= outputHeight) return;

    // Calculate the corresponding input coordinates (nearest neighbor)
    int inX = outX * inputWidth / outputWidth;
    int inY = outY * inputHeight / outputHeight;

    // Calculate indices in input and output images
    int inIndex = inY * inputWidth + inX;
    int outIndex = outY * outputWidth + outX;

    // Scale up the pixel value from the downsampled image to the output image
    output[outIndex] = input[inIndex];
}

void resizeImg(unsigned char* d_img, int imgWidth, int imgHeight, int downsampleFactor){
    const int newWidth = imgWidth / downsampleFactor;
    const int newHeight = imgHeight / downsampleFactor;
    
    // Allocate memory for the downsampled image
    unsigned char* d_img_downsampled;
    unsigned char* h_img_downsampled = (unsigned char*)malloc(newWidth * newHeight * sizeof(unsigned char));
    cudaMalloc(&d_img_downsampled, newWidth * newHeight * sizeof(unsigned char));
    
    // Define block size and grid size for the downsampled image
    dim3 downBlockSize(16, 16);
    dim3 downGridSize((newWidth + downBlockSize.x - 1) / downBlockSize.x, 
                      (newHeight + downBlockSize.y - 1) / downBlockSize.y);

    // Downsample the image
    resizeKernel<<<downGridSize, downBlockSize>>>(d_img, d_img_downsampled, imgWidth, imgHeight, newWidth, newHeight, true);
    // resizeImg(d_img, d_img_downsampled, imgWidth, imgHeight, downsampleFactor, true);
    cudaDeviceSynchronize();

    // Allocate memory for the upsampled image
    unsigned char* d_img_upsampled;
    unsigned char* h_img_upsampled = (unsigned char*)malloc(imgWidth * imgHeight * sizeof(unsigned char));
    cudaMalloc(&d_img_upsampled, imgWidth * imgHeight * sizeof(unsigned char));
    dim3 upBlockSize(16, 16);
    dim3 upGridSize((imgWidth + upBlockSize.x - 1) / upBlockSize.x, 
                    (imgHeight + upBlockSize.y - 1) / upBlockSize.y);

    // Upsample the downsampled image to the original size
    upsampleKernel<<<upGridSize, upBlockSize>>>(d_img_downsampled, d_img_upsampled, newWidth, newHeight, imgWidth, imgHeight);
    // upScaleDisplay(d_img_downsampled, d_img_upsampled, newWidth, newHeight, imgWidth, imgHeight);
    cudaDeviceSynchronize();
    // Copy the downsampled image back to the host
    cudaMemcpy(h_img_upsampled, d_img_upsampled, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
        
    // Save or process the image data as needed...
    stbi_write_png("hollow_circle_down_up_aa.png", imgWidth, imgHeight, 1, h_img_upsampled, imgWidth);
    // Save or process the downsampled image as needed...
    // Free the GPU memory
    cudaFree(d_img_downsampled);
    cudaFree(d_img_upsampled);
    // Free the host memory
    free(h_img_downsampled);
    free(d_img_upsampled);
}