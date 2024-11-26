#include <cstddef>
#include <iostream>
#include <iomanip>

// External symbols for the binary data
extern unsigned char _binary____relu_inode_bin_start[];
extern unsigned char _binary____relu_inode_bin_end[];
extern size_t _binary____relu_inode_bin_size;
static const size_t relu_inode_size = reinterpret_cast<size_t>(&_binary____relu_inode_bin_size);


// Function to validate and print the binary content
void validate_device_binary(unsigned char* device_binary, size_t device_binary_size) {
    std::cout << "Binary content (size: " << device_binary_size << " bytes):\n";
    for (size_t i = 0; i < device_binary_size; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(device_binary[i]) << " ";
        if ((i + 1) % 16 == 0) {
            std::cout << "\n";
        }
    }
    if (device_binary_size % 16 != 0) {
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    unsigned char* relu_inode = _binary____relu_inode_bin_start;

    // size_t relu_inode_size = 100;
    validate_device_binary(relu_inode, relu_inode_size);

    return 0;
}