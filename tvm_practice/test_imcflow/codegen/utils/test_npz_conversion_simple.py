#!/usr/bin/env python3
"""Simple test to verify C++ NPZ conversion logic matches Python.

This test doesn't require cnpy - it just tests the conversion algorithm
by feeding the same raw bytes to both Python and a standalone C++ program.
"""

import os
import sys
import numpy as np
import subprocess
import tempfile

# Add scan_codegen to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_codegen import load_scan_values_from_npz


def create_test_npz(npz_path: str, seed: int = 42):
    """Create a test NPZ file with random data."""
    np.random.seed(seed)
    # Generate 64 random bytes for one IMCE
    scan_data = np.random.randint(0, 256, size=64, dtype=np.uint8)
    np.savez(npz_path, arr_0=scan_data)
    return scan_data


def generate_cpp_test_standalone(cpp_path: str, raw_bytes: np.ndarray):
    """Generate standalone C++ test with hardcoded bytes (no cnpy needed)."""
    
    # Convert raw bytes to C array initializer
    bytes_str = ', '.join(str(b) for b in raw_bytes)
    
    cpp_code = f"""
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

int main() {{
    // Test data: 64 bytes from NPZ file
    uint8_t scan_bytes[64] = {{ {bytes_str} }};
    
    printf("First 16 bytes: ");
    for (int i = 0; i < 16; i++) {{
        printf("%d ", scan_bytes[i]);
    }}
    printf("\\n");
    
    // Allocate memory for scan data (2 packets for 1 IMCE)
    const int num_packets = 2;
    const int bytes_per_packet = 32;
    uint32_t* scan_data = (uint32_t*)malloc(num_packets * bytes_per_packet);
    if (!scan_data) {{
        fprintf(stderr, "Error: Failed to allocate memory\\n");
        return -1;
    }}
    
    // Convert bytes following the same logic as load_scan_values_from_npz
    const int bytes_per_imce = 64;
    uint8_t* imce_bytes = scan_bytes;
    
    // Step 1: Convert 64 bytes to 512-bit string
    char bit_str[513];
    for (int i = 0; i < bytes_per_imce; i++) {{
        for (int b = 0; b < 8; b++) {{
            bit_str[i * 8 + b] = ((imce_bytes[i] >> (7 - b)) & 1) ? '1' : '0';
        }}
    }}
    bit_str[512] = '\\0';
    
    // Step 2: Reverse the entire bit string
    char rev_bit_str[513];
    for (int i = 0; i < 512; i++) {{
        rev_bit_str[i] = bit_str[511 - i];
    }}
    rev_bit_str[512] = '\\0';
    
    // Step 3: Extract reg1 (bits 0-256) and reg0 (bits 256-512)
    char reg1_bits[257], reg0_bits[257];
    memcpy(reg1_bits, &rev_bit_str[0], 256);
    reg1_bits[256] = '\\0';
    memcpy(reg0_bits, &rev_bit_str[256], 256);
    reg0_bits[256] = '\\0';
    
    // Step 4: Convert bit strings to int16 packets
    int16_t* packet_0 = (int16_t*)&scan_data[0 * 8];
    int16_t* packet_1 = (int16_t*)&scan_data[1 * 8];
    
    // Convert reg0_bits to packet_0
    for (int i = 0; i < 16; i++) {{
        char bits_16[17];
        memcpy(bits_16, &reg0_bits[i * 16], 16);
        bits_16[16] = '\\0';
        
        char bits_16_rev[17];
        for (int b = 0; b < 16; b++) {{
            bits_16_rev[b] = bits_16[15 - b];
        }}
        bits_16_rev[16] = '\\0';
        
        int32_t val = 0;
        for (int b = 0; b < 16; b++) {{
            if (bits_16_rev[b] == '1') {{
                val |= (1 << (15 - b));  // Leftmost char is MSB
            }}
        }}
        
        if (val >= 32768) {{
            val -= 65536;
        }}
        packet_0[i] = (int16_t)val;
    }}
    
    // Convert reg1_bits to packet_1
    for (int i = 0; i < 16; i++) {{
        char bits_16[17];
        memcpy(bits_16, &reg1_bits[i * 16], 16);
        bits_16[16] = '\\0';
        
        char bits_16_rev[17];
        for (int b = 0; b < 16; b++) {{
            bits_16_rev[b] = bits_16[15 - b];
        }}
        bits_16_rev[16] = '\\0';
        
        int32_t val = 0;
        for (int b = 0; b < 16; b++) {{
            if (bits_16_rev[b] == '1') {{
                val |= (1 << (15 - b));  // Leftmost char is MSB
            }}
        }}
        
        if (val >= 32768) {{
            val -= 65536;
        }}
        packet_1[i] = (int16_t)val;
    }}
    
    // Print results
    printf("PACKET_0:");
    for (int i = 0; i < 16; i++) {{
        printf(" %d", packet_0[i]);
    }}
    printf("\\n");
    
    printf("PACKET_1:");
    for (int i = 0; i < 16; i++) {{
        printf(" %d", packet_1[i]);
    }}
    printf("\\n");
    
    free(scan_data);
    return 0;
}}
"""
    
    with open(cpp_path, 'w') as f:
        f.write(cpp_code)
    print(f"Generated standalone C++ test: {cpp_path}")


def compile_and_run_cpp(cpp_path: str, exe_path: str):
    """Compile and run C++ test."""
    # Compile
    cmd = ['g++', '-o', exe_path, cpp_path, '-std=c++11']
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("COMPILATION FAILED:")
        print(result.stderr)
        return None, None
    
    print(f"Compiled successfully: {exe_path}")
    
    # Run
    print(f"Running: {exe_path}")
    result = subprocess.run([exe_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("EXECUTION FAILED:")
        print(result.stderr)
        return None, None
    
    print(result.stdout)
    
    # Parse output
    lines = result.stdout.strip().split('\n')
    packet_0 = None
    packet_1 = None
    
    for line in lines:
        if line.startswith('PACKET_0:'):
            packet_0 = [int(x) for x in line.split(':')[1].strip().split()]
        elif line.startswith('PACKET_1:'):
            packet_1 = [int(x) for x in line.split(':')[1].strip().split()]
    
    return packet_0, packet_1


def compare_results(py_packets, cpp_packet_0, cpp_packet_1):
    """Compare Python and C++ results."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if cpp_packet_0 is None or cpp_packet_1 is None:
        print("❌ C++ test failed to produce output")
        return False
    
    if len(py_packets) != 2:
        print(f"❌ Python produced {len(py_packets)} packets, expected 2")
        return False
    
    py_packet_0 = py_packets[0]
    py_packet_1 = py_packets[1]
    
    # Compare packet 0
    print("\nPacket 0:")
    print(f"  Python: {py_packet_0}")
    print(f"  C++:    {cpp_packet_0}")
    
    if py_packet_0 == cpp_packet_0:
        print("  ✓ MATCH")
        packet_0_ok = True
    else:
        print("  ❌ MISMATCH")
        for i, (py_val, cpp_val) in enumerate(zip(py_packet_0, cpp_packet_0)):
            if py_val != cpp_val:
                print(f"    Index {i}: Python={py_val}, C++={cpp_val}")
        packet_0_ok = False
    
    # Compare packet 1
    print("\nPacket 1:")
    print(f"  Python: {py_packet_1}")
    print(f"  C++:    {cpp_packet_1}")
    
    if py_packet_1 == cpp_packet_1:
        print("  ✓ MATCH")
        packet_1_ok = True
    else:
        print("  ❌ MISMATCH")
        for i, (py_val, cpp_val) in enumerate(zip(py_packet_1, cpp_packet_1)):
            if py_val != cpp_val:
                print(f"    Index {i}: Python={py_val}, C++={cpp_val}")
        packet_1_ok = False
    
    print("\n" + "="*60)
    if packet_0_ok and packet_1_ok:
        print("✅ ALL TESTS PASSED")
        print("C++ conversion logic matches Python load_scan_values_from_npz!")
        return True
    else:
        print("❌ TESTS FAILED")
        print("C++ conversion differs from Python!")
        return False


def main():
    """Run the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, 'test_scan.npz')
        cpp_path = os.path.join(tmpdir, 'test_conversion.cpp')
        exe_path = os.path.join(tmpdir, 'test_conversion')
        
        print("="*60)
        print("NPZ CONVERSION TEST (Standalone - No cnpy required)")
        print("="*60)
        
        # Create test NPZ and get raw bytes
        print("\n[1/4] Creating test NPZ file...")
        raw_bytes = create_test_npz(npz_path)
        print(f"Created: {npz_path}")
        print(f"First 16 bytes: {raw_bytes[:16].tolist()}")
        
        # Run Python conversion
        print("\n[2/4] Running Python conversion...")
        py_packets = load_scan_values_from_npz(npz_path, imce_count=1)
        print(f"Python produced {len(py_packets)} packets:")
        print(f"  Packet 0: {py_packets[0]}")
        print(f"  Packet 1: {py_packets[1]}")
        
        # Generate and run C++ test
        print("\n[3/4] Generating and compiling C++ test...")
        generate_cpp_test_standalone(cpp_path, raw_bytes)
        
        print("\n[4/4] Running C++ test...")
        cpp_packet_0, cpp_packet_1 = compile_and_run_cpp(cpp_path, exe_path)
        
        # Compare
        success = compare_results(py_packets, cpp_packet_0, cpp_packet_1)
        
        return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
