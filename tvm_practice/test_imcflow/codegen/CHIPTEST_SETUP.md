# Chip Test Setup and Usage Guide

This guide provides comprehensive documentation for running chip tests on the remote hardware platform, including setup instructions and script usage.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Script Overview](#script-overview)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

Before running chip tests, ensure the following tools are installed:

#### 1. sshpass (for automated password authentication)
```bash
apt-get update
apt-get install -y sshpass
```

**Purpose**: Allows automated SSH/SCP operations without manual password entry.

#### 2. Cross-compilation toolchain (for ARM targets)
```bash
apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

**Purpose**: Required to compile host binaries for ARM architecture (used in chip testing).

**Alternative names**: `linux-aarch64`, `aarch64-linux-gnu`


### Build Requirements

Before running chip tests, the host binary **must be built** with the correct ISA setting:

```bash
cd host_binary_make
./build.sh execute_graph.c <test_name>_evl arm
cd ..
```

---

## Initial Setup

### Remote Server Configuration

The scripts are configured to connect to the following remote server:

| Parameter | Value |
|-----------|-------|
| Host | 147.46.117.99 |
| Port | 1326 |
| User | root |
| Password | root |
| Base Path | /home/root/tvm/tvm_practice/test_imcflow/codegen |

### Verify Connectivity

Test your connection to the remote server:

```bash
sshpass -p "root" ssh -p 1326 root@147.46.117.99 "echo 'Connection successful'"
```

If this command succeeds, you're ready to run chip tests.

---

## Script Overview

### 1. run_chiptest.sh

**Purpose**: Automated end-to-end chip test workflow

**Location**: `/root/project/tvm/tvm_practice/test_imcflow/codegen/run_chiptest.sh`

#### Workflow Steps:
1. **Generate test folder**: Runs `python test.py -k "{test_name} and {input_setting}" -s`
2. **Transfer test folder**: Copies `{test_name}_evl` to remote server
3. **Transfer host binary**: Copies `host_binary_make` directory to remote server
4. **Execute on chip**: Runs `tvm_host_runner` on remote hardware

#### Syntax:
```bash
./run_chiptest.sh <test_name> <input_setting>
```

#### Parameters:
- `test_name`: Name of the test (e.g., `one_relu`, `one_conv_small`)
- `input_setting`: Input data configuration (e.g., `ones`, `random`, `incremental`)

#### Options:
- `-h, --help`: Display help message

#### Examples:
```bash
# Run one_relu test with ones input
./run_chiptest.sh one_relu ones

# Run one_conv_small test with random input
./run_chiptest.sh one_conv_small random

# Display help
./run_chiptest.sh -h
```

---

### 2. transfer_evl.sh

**Purpose**: Transfer test folders or arbitrary directories to remote server

**Location**: `/root/project/tvm/tvm_practice/test_imcflow/codegen/transfer_evl.sh`

#### Usage Modes:

##### Pattern-based transfer (for _evl directories):
```bash
./transfer_evl.sh <pattern> [username]
```

##### Custom path transfer:
```bash
./transfer_evl.sh --path <directory> [username]
./transfer_evl.sh -p <directory> [username]
```

#### Parameters:
- `pattern`: Pattern to match test directories (e.g., `one_*`, `resnet8_*`, `one_relu`)
  - Can be specified with or without `_evl` suffix
  - Use `'*'` to match all _evl directories
- `--path, -p`: Transfer a specific directory (absolute or relative path)
- `username`: Optional username for remote connection (default: `root`)

#### Examples:
```bash
# Transfer all one_*_evl directories
./transfer_evl.sh one_*

# Transfer one_relu_evl directory (both are equivalent)
./transfer_evl.sh one_relu
./transfer_evl.sh one_relu_evl

# Transfer all _evl directories
./transfer_evl.sh '*'

# Transfer specific directory
./transfer_evl.sh --path host_binary_make

# Transfer with custom username
./transfer_evl.sh one_relu myuser
```

#### Available Test Patterns:
Based on MODEL_REGISTRY in test.py:
- `one_*`: one_relu, one_conv_small, one_conv_big, one_mmquant, etc.
- `resnet8_*`: resnet8_subset01-25 variants
- `conv_*`: conv_quant_conv, etc.
- `residual_*`: residual_model, residual_rnd_model

---

## Usage Examples

### Complete Workflow Example

```bash
# 1. Build host binary for ARM target
cd host_binary_make
./build.sh execute_graph.c one_relu_evl arm
cd ..

# 2. Run chip test (generates test, transfers files, executes on remote)
./run_chiptest.sh one_relu ones
```

### Manual Step-by-Step Workflow

If you prefer manual control over each step:

```bash
# 1. Generate test folder
python test.py -k "one_relu and ones" -s

# 2. Build host binary
cd host_binary_make
./build.sh execute_graph.c one_relu_evl arm
cd ..

# 3. Transfer test folder
./transfer_evl.sh one_relu

# 4. Transfer host binary
./transfer_evl.sh --path host_binary_make

# 5. Execute on remote chip
sshpass -p "root" ssh -p 1326 root@147.46.117.99 \
  "cd /home/root/tvm/tvm_practice/test_imcflow/codegen/host_binary_make/build && \
   ./tvm_host_runner one_relu_evl /home/root/tvm/tvm_practice/test_imcflow/codegen"
```

### Running Multiple Tests

```bash
# Test different input settings
./run_chiptest.sh one_relu ones
./run_chiptest.sh one_relu random
./run_chiptest.sh one_relu incremental

# Test different models
./run_chiptest.sh one_conv_small ones
./run_chiptest.sh resnet8_subset01 ones
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. sshpass command not found
**Error**: `sshpass: command not found`

**Solution**:
```bash
sudo apt-get update
sudo apt-get install -y sshpass
```

#### 2. Cross-compilation toolchain missing
**Error**: `aarch64-linux-gnu-gcc: not found` or similar

**Solution**:
```bash
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

#### 3. Test folder not found
**Error**: `Error: No directories found matching pattern 'xxx_evl'`

**Solution**: Ensure you've run `test.py` to generate the test folder first:
```bash
python test.py -k "test_name and input_setting" -s
```

#### 4. SSH connection refused
**Error**: `Connection refused` or `Connection timed out`

**Solution**:
- Verify the remote server is running
- Check network connectivity
- Verify firewall settings allow connections on port 1326

#### 5. Host binary not built
**Error**: Build errors or missing binary

**Solution**: Build the host binary with correct ISA before running tests:
```bash
cd host_binary_make
./build.sh execute_graph.c <test_name>_evl arm
cd ..
```

#### 6. Permission denied on transfer
**Error**: `Permission denied` during scp

**Solution**:
- Verify the password is correct (default: `root`)
- Check write permissions on remote directory
- Ensure the remote directory exists

#### 7. Test execution fails on remote
**Error**: Test fails during execution

**Solution**:
- Check the remote logs for detailed error messages
- Verify the test folder was transferred completely
- Ensure the host binary was built for the correct architecture (ARM)
- Check that all required files are present in the test folder

---

## Additional Notes

### Security Considerations

- The password is stored in plain text in the script for convenience
- For production environments, consider using SSH key-based authentication instead
- Limit access to the scripts on shared systems

### Performance Tips

- Build the host binary once for multiple tests with the same ISA
- Use pattern matching in `transfer_evl.sh` to transfer multiple test folders at once
- Consider keeping a remote cache of test folders that don't change frequently

### Configuration Customization

To modify remote server settings, edit the configuration section in `run_chiptest.sh`:

```bash
REMOTE_HOST="147.46.117.99"
REMOTE_PORT="1326"
REMOTE_USER="root"
REMOTE_PASSWORD="root"
REMOTE_BASE_PATH="/home/root/tvm/tvm_practice/test_imcflow/codegen"
```

---

## Quick Reference

### Common Commands

```bash
# Show help for run_chiptest.sh
./run_chiptest.sh -h

# Show help for transfer_evl.sh
./transfer_evl.sh -h

# Run a complete chip test
./run_chiptest.sh one_relu ones

# Transfer all test folders
./transfer_evl.sh '*'

# Check sshpass installation
which sshpass

# Check cross-compilation toolchain
which aarch64-linux-gnu-gcc

# Test SSH connection
sshpass -p "root" ssh -p 1326 root@147.46.117.99 "echo OK"
```

### File Structure

```
codegen/
├── run_chiptest.sh          # Main chip test automation script
├── transfer_evl.sh          # File transfer utility
├── test.py                  # Test generation script
├── host_binary_make/        # Host binary build directory
│   ├── build.sh             # Build script
│   └── build/               # Build output directory
└── *_evl/                   # Generated test folders
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the help output: `./run_chiptest.sh -h` or `./transfer_evl.sh -h`
3. Verify all prerequisites are installed
4. Check the remote server logs for detailed error messages
