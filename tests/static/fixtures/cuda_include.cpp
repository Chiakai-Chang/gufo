// Negative fixture: Forbidden CUDA header inclusion
#include <cuda_runtime.h>
#include <iostream>

int main() {
    std::cout << "Invalid CUDA header inclusion\n";
    return 0;
}
