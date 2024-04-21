#include "common.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <iostream>

// =================
// Helper Functions
// =================

const int imgWidth = 1024;
const int imgHeight = 1024;
const int centerX = imgWidth / 2;
const int centerY = imgHeight / 2;
const int radius = imgWidth / 4; // for example
const int thickness = 5; // thickness of the hollow part

// Kernel to generate a hollow circle
__global__ void drawHollowCircle(unsigned char* img) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imgWidth || y >= imgHeight) return;

    // Calculate the index in the array
    int idx = y * imgWidth + x;

    // Calculate the distance from the center of the circle
    int dist = sqrtf((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));

    // Determine if the pixel is on the edge of the circle
    if (dist > (radius - thickness) && dist < (radius + thickness)) {
        img[idx] = 0; // Black pixel
    } else {
        img[idx] = 255; // White pixel
    }
}

void printImage(unsigned char *img) {
    for (int y = 0; y < imgHeight; y++) {
        for (int x = 0; x < imgWidth; x++) {
            std::cout << (img[y * imgWidth + x] == 1 ? "1" : " ");
        }
        std::cout << std::endl;
    }
}


// ==============
// Main Function
// ==============

int main(int argc, char** argv) {
    std::cout << "Program Begins" << std::endl;
    // Allocate memory for the image
    size_t imgSize = imgWidth * imgHeight;
    unsigned char* h_img = new unsigned char[imgSize]();
    unsigned char* d_img;

    cudaMalloc(&d_img, imgSize);

    // Define block size and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((imgWidth + blockSize.x - 1) / blockSize.x,
                  (imgHeight + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    drawHollowCircle<<<gridSize, blockSize>>>(d_img);

    // Copy the generated image back to host memory
    cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Define the downsample factor and calculate the new dimensions
    const int downsampleFactor = 16; // Replace with the desired factor
    resizeImg(d_img, imgWidth, imgHeight, downsampleFactor);

    // Free the memory
    delete[] h_img;
    cudaFree(d_img);

    return 0;
}
