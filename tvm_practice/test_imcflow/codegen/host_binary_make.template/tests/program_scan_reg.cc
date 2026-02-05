
#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include <stdint.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cnpy.h>
#include <vector>

#define IMCFLOW_ADDR 2147483648
#define IMCFLOW_LEN 266368
#define INT_ACK_GEN_ADDR 0
#define INT_ACK_GEN_LEN 0
#define IMCFLOW_DEVICE "/dev/uio5"
#define INT_ACK_GEN_DEVICE "/dev/uio4"
#define STATE_REG_IDX 0
#define PC_REG_IDX 2
#define INTR_DONE_REG_IDX 7
#define SET_IDLE_CODE 0
#define SET_RUN_CODE 1
#define SET_PROGRAM_CODE 2
#define INODE_PC_START_P0_ENUM_VAL 0
#define INODE_PC_START_EXTERN_ENUM_VAL 1
#define INODE_NUM 4

// short16 type definition
typedef short short16 __attribute__((ext_vector_type(16)));

extern "C" {
  extern const int32_t _binary_program_scan_reg_imce_0_1_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_1_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_1_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_1_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_2_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_2_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_2_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_2_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_3_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_3_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_3_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_3_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_4_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_4_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_0_4_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_0_4_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_1_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_1_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_1_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_1_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_2_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_2_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_2_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_2_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_3_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_3_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_3_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_3_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_4_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_4_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_1_4_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_1_4_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_1_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_1_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_1_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_1_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_2_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_2_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_2_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_2_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_3_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_3_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_3_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_3_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_4_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_4_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_2_4_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_2_4_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_1_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_1_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_1_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_1_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_2_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_2_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_2_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_2_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_3_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_3_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_3_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_3_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_4_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_4_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_imce_3_4_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_imce_3_4_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_0_0_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_0_0_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_0_0_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_0_0_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_1_0_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_1_0_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_1_0_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_1_0_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_2_0_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_2_0_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_2_0_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_2_0_scan_reg_policy_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_3_0_imem_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_3_0_imem_bin_end[];
  extern const int32_t _binary_program_scan_reg_inode_3_0_scan_reg_policy_bin_start[];
  extern const int32_t _binary_program_scan_reg_inode_3_0_scan_reg_policy_bin_end[];
}


static inline void enable_imcflow_interrupt(int fd)
{
  uint32_t info = 1;
  ssize_t nb = write(fd, &info, sizeof(info));
  if (nb != (ssize_t)sizeof(info)) {
    perror("write failed");
    close(fd);
    exit(1);
  }
}

static inline void wait_imcflow_interrupt(int fd)
{
  uint32_t info;
  ssize_t nb = read(fd, &info, sizeof(info));
  (void)nb;
}

static inline void generate_ack(uint32_t* int_ack_gen)
{
  int_ack_gen[0] = 0b1;
}

// Poll until ImcFlow returns to IDLE state
static void wait_for_idle(volatile uint32_t* npu_pointer)
{
  while (npu_pointer[STATE_REG_IDX] != SET_IDLE_CODE) {
    // Busy wait
  }
}


#ifdef __cplusplus
extern "C"
#endif
int32_t program_scan_reg(const char* file_name) {
    // Check if file_name is provided
    if (file_name == NULL || file_name[0] == '\0') {
        fprintf(stderr, "Error: NPZ directory path is required for program_scan_reg\n");
        fprintf(stderr, "Usage: program_scan_reg <npz_directory_path>\n");
        return -1;
    }

    // Strip surrounding quotes from file_name if present
    std::string file_name_str(file_name);
    if (file_name_str.size() >= 2 && file_name_str.front() == '"' && file_name_str.back() == '"') {
        file_name_str = file_name_str.substr(1, file_name_str.size() - 2);
    }
    const char* actual_file_name = file_name_str.c_str();

    fprintf(stderr, "Starting program_scan_reg from directory: %s\n", actual_file_name);

    // Allocate memory for scan data (32 packets × 32 bytes = 1024 bytes)
    const int num_packets = 32;
    const int bytes_per_packet = 32;
    uint32_t* scan_data = (uint32_t*)malloc(num_packets * bytes_per_packet);
    if (!scan_data) {
        fprintf(stderr, "Error: Failed to allocate memory for scan data\n");
        return -1;
    }

    // Convert NPZ files to scan packets
    // Each IMCE has its own NPZ file with 64 bytes → 2 packets (32 bytes each)
    // 16 IMCEs × 2 packets = 32 packets total
    const int imce_count = 16;
    const int bytes_per_imce = 64;

    // Convert NPZ bytes to scan packets following the same logic as load_scan_values_from_npz
    // For each IMCE: load NPZ file, 64 bytes → 512 bits → bit-reverse → split into reg0/reg1 → 2 packets
    fprintf(stderr, "Loading and converting NPZ data with bit reversal...\n");
    
    for (int imce_idx = 0; imce_idx < imce_count; imce_idx++) {
        // Calculate IMCE coordinates: h=0-3, w=1-4
        int h = imce_idx / 4;
        int w = imce_idx % 4;
        std::string imce_filename = std::string(actual_file_name) + "/imce_" + std::to_string(h) + "_" + std::to_string(w) + ".npz";
        
        // Load IMCE NPZ file
        cnpy::npz_t imce_npz;
        try {
            imce_npz = cnpy::npz_load(imce_filename);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error loading NPZ file %s: %s\n", imce_filename.c_str(), e.what());
            free(scan_data);
            return -1;
        }

        if (imce_npz.find("arr_0") == imce_npz.end()) {
            fprintf(stderr, "Error: 'arr_0' not found in NPZ file %s\n", imce_filename.c_str());
            free(scan_data);
            return -1;
        }

        cnpy::NpyArray imce_arr = imce_npz["arr_0"];
        uint8_t* imce_bytes = imce_arr.data<uint8_t>();
        size_t imce_bytes_size = imce_arr.num_bytes();

        if (imce_bytes_size != bytes_per_imce) {
            fprintf(stderr, "Error: Expected %d bytes in NPZ file %s, got %zu bytes\n", 
                    bytes_per_imce, imce_filename.c_str(), imce_bytes_size);
            free(scan_data);
            return -1;
        }
        
        // Step 1: Convert 64 bytes to 512-bit string
        char bit_str[513];  // 512 bits + null terminator
        for (int i = 0; i < bytes_per_imce; i++) {
            for (int b = 0; b < 8; b++) {
                bit_str[i * 8 + b] = ((imce_bytes[i] >> (7 - b)) & 1) ? '1' : '0';
            }
        }
        bit_str[512] = '\0';
        
        // Step 2: Reverse the entire bit string
        char rev_bit_str[513];
        for (int i = 0; i < 512; i++) {
            rev_bit_str[i] = bit_str[511 - i];
        }
        rev_bit_str[512] = '\0';
        
        // Step 3: Extract reg1 (bits 0-256) and reg0 (bits 256-512)
        // reg1 corresponds to packet 1 (bytes 0-31 after reversal)
        // reg0 corresponds to packet 0 (bytes 32-63 after reversal)
        char reg1_bits[257], reg0_bits[257];
        memcpy(reg1_bits, &rev_bit_str[0], 256);
        reg1_bits[256] = '\0';
        memcpy(reg0_bits, &rev_bit_str[256], 256);
        reg0_bits[256] = '\0';
        
        // Step 4: Convert bit strings to int16 packets
        // Each packet has 16 int16 values
        int16_t* packet_0 = (int16_t*)&scan_data[(imce_idx * 2 + 0) * 8];  // 32 bytes = 8 uint32_t = 16 int16
        int16_t* packet_1 = (int16_t*)&scan_data[(imce_idx * 2 + 1) * 8];
        
        // Convert reg0_bits to packet_0
        for (int i = 0; i < 16; i++) {
            // Extract 16 bits for this int16
            char bits_16[17];
            memcpy(bits_16, &reg0_bits[i * 16], 16);
            bits_16[16] = '\0';
            
            // Reverse bits for little-endian int16 interpretation
            char bits_16_rev[17];
            for (int b = 0; b < 16; b++) {
                bits_16_rev[b] = bits_16[15 - b];
            }
            bits_16_rev[16] = '\0';
            
            // Convert binary string to int16 (treat leftmost char as MSB)
            int32_t val = 0;
            for (int b = 0; b < 16; b++) {
                if (bits_16_rev[b] == '1') {
                    val |= (1 << (15 - b));  // Leftmost char is MSB (bit 15)
                }
            }
            
            // Handle signed conversion (2's complement)
            if (val >= 32768) {
                val -= 65536;
            }
            packet_0[i] = (int16_t)val;
        }
        
        // Convert reg1_bits to packet_1
        for (int i = 0; i < 16; i++) {
            // Extract 16 bits for this int16
            char bits_16[17];
            memcpy(bits_16, &reg1_bits[i * 16], 16);
            bits_16[16] = '\0';
            
            // Reverse bits for little-endian int16 interpretation
            char bits_16_rev[17];
            for (int b = 0; b < 16; b++) {
                bits_16_rev[b] = bits_16[15 - b];
            }
            bits_16_rev[16] = '\0';
            
            // Convert binary string to int16 (treat leftmost char as MSB)
            int32_t val = 0;
            for (int b = 0; b < 16; b++) {
                if (bits_16_rev[b] == '1') {
                    val |= (1 << (15 - b));  // Leftmost char is MSB (bit 15)
                }
            }
            
            // Handle signed conversion (2's complement)
            if (val >= 32768) {
                val -= 65536;
            }
            packet_1[i] = (int16_t)val;
        }
    }
    
    fprintf(stderr, "Loaded and converted %d scan packets from NPZ files\n", num_packets);


    uint32_t* npu_ptr = (uint32_t*)IMCFLOW_ADDR;
    uint32_t* int_ack_ptr = (uint32_t*)INT_ACK_GEN_ADDR;


    // Transfer policy tables and instruction memory from binary files
    fprintf(stderr, "Transferring policy tables and instruction memory to NPU:\n");
    // Transfer inode_0_0_imem (160 bytes at 0x80)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_0_0_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_0_0_imem_bin_end - _binary_program_scan_reg_inode_0_0_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(128 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_0_0_imem: 0x%x (%zu bytes)\n", 128, len * 4);
    }
    // Transfer inode_0_0_scan_reg_policy (160 bytes at 0x480)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_0_0_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_0_0_scan_reg_policy_bin_end - _binary_program_scan_reg_inode_0_0_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(1152 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_0_0_scan_reg_policy: 0x%x (%zu bytes)\n", 1152, len * 4);
    }
    // Transfer imce_0_1_scan_reg_policy (160 bytes at 0x520)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_1_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_1_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_0_1_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(1312 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_1_scan_reg_policy: 0x%x (%zu bytes)\n", 1312, len * 4);
    }
    // Transfer imce_0_2_scan_reg_policy (128 bytes at 0x5c0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_2_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_2_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_0_2_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(1472 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_2_scan_reg_policy: 0x%x (%zu bytes)\n", 1472, len * 4);
    }
    // Transfer imce_0_3_scan_reg_policy (96 bytes at 0x640)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_3_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_3_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_0_3_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(1600 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_3_scan_reg_policy: 0x%x (%zu bytes)\n", 1600, len * 4);
    }
    // Transfer imce_0_4_scan_reg_policy (64 bytes at 0x6a0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_4_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_4_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_0_4_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(1696 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_4_scan_reg_policy: 0x%x (%zu bytes)\n", 1696, len * 4);
    }
    // Transfer imce_0_1_imem (192 bytes at 0x7e0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_1_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_1_imem_bin_end - _binary_program_scan_reg_imce_0_1_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(2016 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_1_imem: 0x%x (%zu bytes)\n", 2016, len * 4);
    }
    // Transfer imce_0_2_imem (192 bytes at 0x8a0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_2_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_2_imem_bin_end - _binary_program_scan_reg_imce_0_2_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(2208 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_2_imem: 0x%x (%zu bytes)\n", 2208, len * 4);
    }
    // Transfer imce_0_3_imem (192 bytes at 0x960)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_3_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_3_imem_bin_end - _binary_program_scan_reg_imce_0_3_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(2400 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_3_imem: 0x%x (%zu bytes)\n", 2400, len * 4);
    }
    // Transfer imce_0_4_imem (192 bytes at 0xa20)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_0_4_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_0_4_imem_bin_end - _binary_program_scan_reg_imce_0_4_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(2592 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_0_4_imem: 0x%x (%zu bytes)\n", 2592, len * 4);
    }
    // Transfer inode_1_0_imem (160 bytes at 0x10480)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_1_0_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_1_0_imem_bin_end - _binary_program_scan_reg_inode_1_0_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(66688 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_1_0_imem: 0x%x (%zu bytes)\n", 66688, len * 4);
    }
    // Transfer inode_1_0_scan_reg_policy (160 bytes at 0x10880)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_1_0_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_1_0_scan_reg_policy_bin_end - _binary_program_scan_reg_inode_1_0_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(67712 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_1_0_scan_reg_policy: 0x%x (%zu bytes)\n", 67712, len * 4);
    }
    // Transfer imce_1_1_scan_reg_policy (160 bytes at 0x10920)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_1_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_1_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_1_1_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(67872 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_1_scan_reg_policy: 0x%x (%zu bytes)\n", 67872, len * 4);
    }
    // Transfer imce_1_2_scan_reg_policy (128 bytes at 0x109c0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_2_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_2_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_1_2_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(68032 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_2_scan_reg_policy: 0x%x (%zu bytes)\n", 68032, len * 4);
    }
    // Transfer imce_1_3_scan_reg_policy (96 bytes at 0x10a40)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_3_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_3_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_1_3_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(68160 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_3_scan_reg_policy: 0x%x (%zu bytes)\n", 68160, len * 4);
    }
    // Transfer imce_1_4_scan_reg_policy (64 bytes at 0x10aa0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_4_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_4_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_1_4_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(68256 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_4_scan_reg_policy: 0x%x (%zu bytes)\n", 68256, len * 4);
    }
    // Transfer imce_1_1_imem (192 bytes at 0x10be0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_1_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_1_imem_bin_end - _binary_program_scan_reg_imce_1_1_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(68576 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_1_imem: 0x%x (%zu bytes)\n", 68576, len * 4);
    }
    // Transfer imce_1_2_imem (192 bytes at 0x10ca0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_2_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_2_imem_bin_end - _binary_program_scan_reg_imce_1_2_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(68768 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_2_imem: 0x%x (%zu bytes)\n", 68768, len * 4);
    }
    // Transfer imce_1_3_imem (192 bytes at 0x10d60)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_3_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_3_imem_bin_end - _binary_program_scan_reg_imce_1_3_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(68960 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_3_imem: 0x%x (%zu bytes)\n", 68960, len * 4);
    }
    // Transfer imce_1_4_imem (192 bytes at 0x10e20)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_1_4_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_1_4_imem_bin_end - _binary_program_scan_reg_imce_1_4_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(69152 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_1_4_imem: 0x%x (%zu bytes)\n", 69152, len * 4);
    }
    // Transfer inode_2_0_imem (160 bytes at 0x20880)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_2_0_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_2_0_imem_bin_end - _binary_program_scan_reg_inode_2_0_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(133248 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_2_0_imem: 0x%x (%zu bytes)\n", 133248, len * 4);
    }
    // Transfer inode_2_0_scan_reg_policy (160 bytes at 0x20c80)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_2_0_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_2_0_scan_reg_policy_bin_end - _binary_program_scan_reg_inode_2_0_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(134272 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_2_0_scan_reg_policy: 0x%x (%zu bytes)\n", 134272, len * 4);
    }
    // Transfer imce_2_1_scan_reg_policy (160 bytes at 0x20d20)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_1_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_1_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_2_1_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(134432 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_1_scan_reg_policy: 0x%x (%zu bytes)\n", 134432, len * 4);
    }
    // Transfer imce_2_2_scan_reg_policy (128 bytes at 0x20dc0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_2_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_2_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_2_2_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(134592 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_2_scan_reg_policy: 0x%x (%zu bytes)\n", 134592, len * 4);
    }
    // Transfer imce_2_3_scan_reg_policy (96 bytes at 0x20e40)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_3_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_3_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_2_3_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(134720 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_3_scan_reg_policy: 0x%x (%zu bytes)\n", 134720, len * 4);
    }
    // Transfer imce_2_4_scan_reg_policy (64 bytes at 0x20ea0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_4_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_4_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_2_4_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(134816 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_4_scan_reg_policy: 0x%x (%zu bytes)\n", 134816, len * 4);
    }
    // Transfer imce_2_1_imem (192 bytes at 0x20fe0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_1_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_1_imem_bin_end - _binary_program_scan_reg_imce_2_1_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(135136 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_1_imem: 0x%x (%zu bytes)\n", 135136, len * 4);
    }
    // Transfer imce_2_2_imem (192 bytes at 0x210a0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_2_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_2_imem_bin_end - _binary_program_scan_reg_imce_2_2_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(135328 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_2_imem: 0x%x (%zu bytes)\n", 135328, len * 4);
    }
    // Transfer imce_2_3_imem (192 bytes at 0x21160)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_3_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_3_imem_bin_end - _binary_program_scan_reg_imce_2_3_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(135520 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_3_imem: 0x%x (%zu bytes)\n", 135520, len * 4);
    }
    // Transfer imce_2_4_imem (192 bytes at 0x21220)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_2_4_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_2_4_imem_bin_end - _binary_program_scan_reg_imce_2_4_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(135712 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_2_4_imem: 0x%x (%zu bytes)\n", 135712, len * 4);
    }
    // Transfer inode_3_0_imem (192 bytes at 0x30c80)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_3_0_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_3_0_imem_bin_end - _binary_program_scan_reg_inode_3_0_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(199808 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_3_0_imem: 0x%x (%zu bytes)\n", 199808, len * 4);
    }
    // Transfer inode_3_0_scan_reg_policy (160 bytes at 0x31080)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_inode_3_0_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_inode_3_0_scan_reg_policy_bin_end - _binary_program_scan_reg_inode_3_0_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(200832 / 4) + i] = src[i];
        }
        fprintf(stderr, "  inode_3_0_scan_reg_policy: 0x%x (%zu bytes)\n", 200832, len * 4);
    }
    // Transfer imce_3_1_scan_reg_policy (160 bytes at 0x31120)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_1_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_1_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_3_1_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(200992 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_1_scan_reg_policy: 0x%x (%zu bytes)\n", 200992, len * 4);
    }
    // Transfer imce_3_2_scan_reg_policy (128 bytes at 0x311c0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_2_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_2_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_3_2_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(201152 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_2_scan_reg_policy: 0x%x (%zu bytes)\n", 201152, len * 4);
    }
    // Transfer imce_3_3_scan_reg_policy (96 bytes at 0x31240)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_3_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_3_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_3_3_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(201280 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_3_scan_reg_policy: 0x%x (%zu bytes)\n", 201280, len * 4);
    }
    // Transfer imce_3_4_scan_reg_policy (64 bytes at 0x312a0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_4_scan_reg_policy_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_4_scan_reg_policy_bin_end - _binary_program_scan_reg_imce_3_4_scan_reg_policy_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(201376 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_4_scan_reg_policy: 0x%x (%zu bytes)\n", 201376, len * 4);
    }
    // Transfer imce_3_1_imem (192 bytes at 0x313e0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_1_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_1_imem_bin_end - _binary_program_scan_reg_imce_3_1_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(201696 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_1_imem: 0x%x (%zu bytes)\n", 201696, len * 4);
    }
    // Transfer imce_3_2_imem (192 bytes at 0x314a0)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_2_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_2_imem_bin_end - _binary_program_scan_reg_imce_3_2_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(201888 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_2_imem: 0x%x (%zu bytes)\n", 201888, len * 4);
    }
    // Transfer imce_3_3_imem (192 bytes at 0x31560)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_3_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_3_imem_bin_end - _binary_program_scan_reg_imce_3_3_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(202080 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_3_imem: 0x%x (%zu bytes)\n", 202080, len * 4);
    }
    // Transfer imce_3_4_imem (192 bytes at 0x31620)
    {
        const uint32_t* src = (const uint32_t*)_binary_program_scan_reg_imce_3_4_imem_bin_start;
        size_t len = (size_t)(_binary_program_scan_reg_imce_3_4_imem_bin_end - _binary_program_scan_reg_imce_3_4_imem_bin_start);
        for (size_t i = 0; i < len; i++) {
            npu_ptr[(202272 / 4) + i] = src[i];
        }
        fprintf(stderr, "  imce_3_4_imem: 0x%x (%zu bytes)\n", 202272, len * 4);
    }


    // Transfer scan data to each INode's memory region
    fprintf(stderr, "Transferring scan data to NPU:\n");
    uint32_t* scan_data_ptr = scan_data;
    uint32_t* npu_scan_base;
    // Transfer scan data for inode_0_0 (packets 0-7)
    fprintf(stderr, "  inode_0_0: 0x%x (8 packets)\n", 1760);
    npu_scan_base = &npu_ptr[1760 / 4];
    for (int i = 0; i < 8; i++) {
        // Each packet is 32 bytes = 8 uint32_t
        for (int j = 0; j < 8; j++) {
            npu_scan_base[(i - 0) * 8 + j] = scan_data_ptr[i * 8 + j];
        }
    }
    // Transfer scan data for inode_1_0 (packets 8-15)
    fprintf(stderr, "  inode_1_0: 0x%x (8 packets)\n", 68320);
    npu_scan_base = &npu_ptr[68320 / 4];
    for (int i = 8; i < 16; i++) {
        // Each packet is 32 bytes = 8 uint32_t
        for (int j = 0; j < 8; j++) {
            npu_scan_base[(i - 8) * 8 + j] = scan_data_ptr[i * 8 + j];
        }
    }
    // Transfer scan data for inode_2_0 (packets 16-23)
    fprintf(stderr, "  inode_2_0: 0x%x (8 packets)\n", 134880);
    npu_scan_base = &npu_ptr[134880 / 4];
    for (int i = 16; i < 24; i++) {
        // Each packet is 32 bytes = 8 uint32_t
        for (int j = 0; j < 8; j++) {
            npu_scan_base[(i - 16) * 8 + j] = scan_data_ptr[i * 8 + j];
        }
    }
    // Transfer scan data for inode_3_0 (packets 24-31)
    fprintf(stderr, "  inode_3_0: 0x%x (8 packets)\n", 201440);
    npu_scan_base = &npu_ptr[201440 / 4];
    for (int i = 24; i < 32; i++) {
        // Each packet is 32 bytes = 8 uint32_t
        for (int j = 0; j < 8; j++) {
            npu_scan_base[(i - 24) * 8 + j] = scan_data_ptr[i * 8 + j];
        }
    }

    // Set PC and execute policy update phase
    fprintf(stderr, "Executing policy update phase\n");
    for (int i = 0; i < INODE_NUM; i++) {
        npu_ptr[PC_REG_IDX + i] = (INODE_PC_START_EXTERN_ENUM_VAL << 30);
    }
        npu_ptr[STATE_REG_IDX] = SET_PROGRAM_CODE;
    wait_for_idle(npu_ptr);
    npu_ptr[INTR_DONE_REG_IDX] = 1;

    // Execute scan register programming phase
    fprintf(stderr, "Executing scan register programming phase\n");
    for (int i = 0; i < INODE_NUM; i++) {
        npu_ptr[PC_REG_IDX + i] = (INODE_PC_START_P0_ENUM_VAL << 30);
    }
        npu_ptr[STATE_REG_IDX] = SET_RUN_CODE;
    wait_for_idle(npu_ptr);
    npu_ptr[INTR_DONE_REG_IDX] = 1;

    // Cleanup
    free(scan_data);

    // baremetal: no cleanup


    fprintf(stderr, "program_scan_reg completed successfully\n");
    return 0;
}
