# IMCE Code Generation Documentation

This document describes how IMCE (In-Memory Computing Engine) code is generated from Relay operations through the handler and code block system.

## Architecture Overview

The code generation pipeline consists of two main layers:

1. **Operation Handlers** (`imce_operation_handlers.py`): High-level handlers that process Relay operations and create appropriate code blocks
2. **Code Blocks** (`imce_codeblock.py`): Low-level code generation units that produce actual C++ intrinsic calls

## Operation Handlers

Operation handlers inherit from `OperationHandler` and are responsible for:
- Detecting which Relay operations they can handle
- Creating appropriate code blocks for different execution phases (INIT, EXEC)
- Managing the relationship between operations (e.g., post-ops in convolution)

### Handler Priority System

Handlers are registered with priorities (lower number = higher priority):
- **Priority 0**: CompositeHandler - highest priority
- **Priority 10**: All other operation handlers

---

## 1. CompositeHandler

**Purpose**: Handles composite function calls that wrap sequences of operations.

**Priority**: 0 (highest)

**Handles**: Composite functions with `"Composite"` attribute

**Code Generation**:
- Sets composite context (`composite_id`)
- Visits the body of composite function
- Visits arguments
- why visit arguments last? it was post DFS.

**Generated Code**: None (orchestration only)

---

## 2. ConvHandler

**Purpose**: Handles quantized convolution operations (`nn.imcflow_qconv`).

**Priority**: 10

**Handles**: `nn.imcflow_qconv` operations

**Generated Code Blocks**:

### INIT Phase:
1. **RecvConstBlock** for config register
   ```c
   // config write
   __builtin_IMCE_RECV_CFG(fifo_id);
   ```

### EXEC Phase:
2. **ConvBlock** - Main convolution execution with nested loop structure
This block assume input will be arrived at QINPUT_CONV_LAYOUT order
which is [N, ceil(IC/256), H, W, IB, 8] int32

**Example Generated Code**:
```c
// generate: conv exec0
// outer_loop(iterate row offset) - row_group0
for (int i0 = 0; i0 < row_count; i0++) {
  // inner_loop(iterate col offset. load inputs) - col_group0
  for (int i1 = 0; i1 < col_count; i1++) {
    // load_block
    for (int i2 = 0; i2 < recv_count; i2++) {
      __builtin_IMCE_LOAD_LB(fifo_id); // tensor_edge_info
      __builtin_IMCE_LOAD_LB(fifo_id); // tensor_edge_info
      __builtin_IMCE_LOAD_LB(fifo_id); // tensor_edge_info
      __builtin_IMCE_LOAD_LB(fifo_id); // tensor_edge_info
    }
    __builtin_IMCE_STEP();
    
    var0 = __builtin_IMCE_GET_CREG((short)0);
    var1 = __builtin_IMCE_GET_CREG((short)1);
    var2 = __builtin_IMCE_GET_CREG((short)2);
    var3 = __builtin_IMCE_GET_CREG((short)3);
    
    // Post-ops would be inserted here
    
    __builtin_IMCE_SEND(address, var0, fifo_id, 0);
    __builtin_IMCE_SEND(address, var1, fifo_id, 0);
    __builtin_IMCE_SEND(address, var2, fifo_id, 0);
    __builtin_IMCE_SEND(address, var3, fifo_id, 0);
  }
}
// endgenerate: conv exec0
```

---

## 3. AddHandler

**Purpose**: Handles element-wise addition operations.

**Priority**: 10

**Handles**: `add` operations (must be inside composite)

**Generated Code Block**: **AddBlock** (as post-op to ConvBlock)
This block add psum into conv block ouptuts.
send order of two input edges should be same. 
In other words, they have same layout. 

**Example Generated Code**:
```c
var_out0 = __builtin_IMCE_ADD(var_in0, var_in1, 15);
var_out1 = __builtin_IMCE_ADD(var_in2, var_in3, 15);
var_out2 = __builtin_IMCE_ADD(var_in4, var_in5, 15);
var_out3 = __builtin_IMCE_ADD(var_in6, var_in7, 15);
```

---

## 4. MultHandler

**Purpose**: Handles element-wise multiplication operations.

**Priority**: 10

**Handles**: `multiply` operations (can be standalone or in composite)

**Generated Code Blocks**:

### INIT Phase (for constant operands):
```c
// mult const
var_const = __builtin_IMCE_RECV(fifo_id);
```

### EXEC Phase:
- **MultlBlock** as post-op (if in composite)
- **RecvSendWrapper** wrapped MultlBlock (if standalone)

**Example Generated Code (standalone)**:
```c
// call_created_loop
for (int i0 = 0; i0 < count; i0++) {
  var_in0 = __builtin_IMCE_RECV(fifo_id);
  var_in1 = __builtin_IMCE_RECV(fifo_id);
  var_in2 = __builtin_IMCE_RECV(fifo_id);
  var_in3 = __builtin_IMCE_RECV(fifo_id);
  
  var_out0 = __builtin_IMCE_MULTL(var_in0, var_const0, 15);
  var_out1 = __builtin_IMCE_MULTL(var_in1, var_const1, 15);
  var_out2 = __builtin_IMCE_MULTL(var_in2, var_const2, 15);
  var_out3 = __builtin_IMCE_MULTL(var_in3, var_const3, 15);
  
  __builtin_IMCE_SEND(address, var_out0, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out1, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out2, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out3, fifo_id, 0);
}
```

---

## 5. DivideHandler

**Purpose**: Handles element-wise division operations.

**Priority**: 10

**Handles**: `divide` operations (must be inside composite)

**Generated Code Block**: **DivBlock** (as post-op to ConvBlock)

**Example Generated Code**:
```c
var_out0 = __builtin_IMCE_DIV(var_in0, var_in1, 15);
var_out1 = __builtin_IMCE_DIV(var_in2, var_in3, 15);
var_out2 = __builtin_IMCE_DIV(var_in4, var_in5, 15);
var_out3 = __builtin_IMCE_DIV(var_in6, var_in7, 15);
```

---

## 6. ConcatHandler

**Purpose**: Handles tensor concatenation operations.

**Priority**: 10

**Handles**: `concatenate` operations

**Generated Code Block**: **ConcatBlock** (as post-op to ConvBlock)

**Example Generated Code**:
```c
var_ext0 = __builtin_IMCE_RECV(fifo_id);
var_out0 = __builtin_IMCE_OR(var_internal0, var_ext0, 15);
var_ext1 = __builtin_IMCE_RECV(fifo_id);
var_out1 = __builtin_IMCE_OR(var_internal1, var_ext1, 15);
var_ext2 = __builtin_IMCE_RECV(fifo_id);
var_out2 = __builtin_IMCE_OR(var_internal2, var_ext2, 15);
var_ext3 = __builtin_IMCE_RECV(fifo_id);
var_out3 = __builtin_IMCE_OR(var_internal3, var_ext3, 15);
```

---

## 7. SplitHandler

**Purpose**: Handles tensor split operations.

**Priority**: 10

**Handles**: `split` operations

**Generated Code Block**: **SplitBlock**
Split is handled by send inst, NoC and policy. just check multicast consistency

**Example Generated Code**: None

---

## 8. MinMaxQuantizeHandler

**Purpose**: Handles min-max quantization operations (`qnn.imcflow_min_max_quantize`).

**Priority**: 10

**Handles**: `qnn.imcflow_min_max_quantize` operations

**Generated Code Blocks**:

### INIT Phase:
```c
// min write
__builtin_IMCE_RECV_MIN(fifo_id);

// max write
__builtin_IMCE_RECV_MAX(fifo_id);
```

### EXEC Phase:
**MinmaxQuantBlock** (as post-op or standalone)

**Example Generated Code (as post-op)**:
```c
__builtin_IMCE_MM_QUANT(var_in0, 0, 15, 0);
__builtin_IMCE_MM_QUANT(var_in1, 0, 15, 1);
__builtin_IMCE_MM_QUANT(var_in2, 0, 15, 2);
__builtin_IMCE_MM_QUANT(var_in3, 0, 15, 3);

var_out0 = __builtin_IMCE_GET_QREG(0);
var_out1 = __builtin_IMCE_GET_QREG(1);
var_out2 = __builtin_IMCE_GET_QREG(2);
var_out3 = __builtin_IMCE_GET_QREG(3);
```

**Example Generated Code (standalone)**:
```c
// call_created_loop
for (int i0 = 0; i0 < count; i0++) {
  var_in0 = __builtin_IMCE_RECV(fifo_id);
  var_in1 = __builtin_IMCE_RECV(fifo_id);
  var_in2 = __builtin_IMCE_RECV(fifo_id);
  var_in3 = __builtin_IMCE_RECV(fifo_id);
  
  __builtin_IMCE_MM_QUANT(var_in0, 0, 15, 0);
  __builtin_IMCE_MM_QUANT(var_in1, 0, 15, 1);
  __builtin_IMCE_MM_QUANT(var_in2, 0, 15, 2);
  __builtin_IMCE_MM_QUANT(var_in3, 0, 15, 3);
  
  var_out0 = __builtin_IMCE_GET_QREG(0);
  var_out1 = __builtin_IMCE_GET_QREG(1);
  var_out2 = __builtin_IMCE_GET_QREG(2);
  var_out3 = __builtin_IMCE_GET_QREG(3);
  
  __builtin_IMCE_SEND(address, var_out0, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out1, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out2, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out3, fifo_id, 0);
}
```

---

## 9. ReLUHandler

**Purpose**: Handles ReLU activation operations.

**Priority**: 10

**Handles**: `nn.relu` operations

**Generated Code Block**: **ReLUBlock** (as post-op or standalone)

**Example Generated Code (as post-op)**:
```c
var_out0 = __builtin_IMCE_MAXI(var_in0, 0);
var_out1 = __builtin_IMCE_MAXI(var_in1, 0);
var_out2 = __builtin_IMCE_MAXI(var_in2, 0);
var_out3 = __builtin_IMCE_MAXI(var_in3, 0);
```

**Example Generated Code (standalone with RECV/SEND)**:
```c
// call_created_loop
for (int i0 = 0; i0 < count; i0++) {
  var_in0 = __builtin_IMCE_RECV(fifo_id);
  var_in1 = __builtin_IMCE_RECV(fifo_id);
  var_in2 = __builtin_IMCE_RECV(fifo_id);
  var_in3 = __builtin_IMCE_RECV(fifo_id);
  
  var_out0 = __builtin_IMCE_MAXI(var_in0, 0);
  var_out1 = __builtin_IMCE_MAXI(var_in1, 0);
  var_out2 = __builtin_IMCE_MAXI(var_in2, 0);
  var_out3 = __builtin_IMCE_MAXI(var_in3, 0);
  
  __builtin_IMCE_SEND(address, var_out0, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out1, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out2, fifo_id, 0);
  __builtin_IMCE_SEND(address, var_out3, fifo_id, 0);
}
```

---

## 10. BiasAddHandler

**Purpose**: Handles bias addition operations (currently disabled).

**Priority**: 10

**Handles**: `nn.bias_add` operations (currently returns False)

**Status**: Disabled

---

## 11. BatchNormHandler

**Purpose**: Handles fused batch normalization operations.

**Priority**: 10

**Handles**: `imcflow.fused_batch_norm` operations

**Generated Code Blocks**:

### INIT Phase:
```c
// fused_scale write
var_scale = __builtin_IMCE_RECV(fifo_id);

// fused_bias write
var_bias = __builtin_IMCE_RECV(fifo_id);
```

### EXEC Phase:
**BatchNormBlock** (as post-op to ConvBlock)

**Example Generated Code**:
```c
var_out0 = __builtin_IMCE_MULTL(var_data0, var_scale0, 15);
var_out0 = __builtin_IMCE_ADD(var_out0, var_bias0, 15);

var_out1 = __builtin_IMCE_MULTL(var_data1, var_scale1, 15);
var_out1 = __builtin_IMCE_ADD(var_out1, var_bias1, 15);

var_out2 = __builtin_IMCE_MULTL(var_data2, var_scale2, 15);
var_out2 = __builtin_IMCE_ADD(var_out2, var_bias2, 15);

var_out3 = __builtin_IMCE_MULTL(var_data3, var_scale3, 15);
var_out3 = __builtin_IMCE_ADD(var_out3, var_bias3, 15);
```

---

## 12. NuQuantizeHandler

**Purpose**: Handles NU quantization operations (currently disabled).

**Priority**: 10

**Handles**: `qnn.imcflow_nu_quantize` operations

**Generated Code**: None (placeholder - does nothing)

---

## Code Block Details

### Base Classes

#### ImceCodeBlock
- Base class for all IMCE code blocks
- Wraps content with annotation comments if provided
- Abstract `_content()` method must be implemented

#### ImceCallCodeBlock
- Base for code blocks that process Relay calls
- Manages input/output edges
- Tracks previous operation for post-op chaining
- Default `num_blocks = 4` (bitplane parallelism)

### Utility Classes

#### LoadLBBlock
**Purpose**: Load data into line buffer from FIFO

**Example**:
```c
// load_block
for (int i0 = 0; i0 < count; i0++) {
  __builtin_IMCE_LOAD_LB(fifo_id); // edge_info
  __builtin_IMCE_LOAD_LB(fifo_id); // edge_info
  __builtin_IMCE_LOAD_LB(fifo_id); // edge_info
  __builtin_IMCE_LOAD_LB(fifo_id); // edge_info
}
```

#### RecvConstBlock
**Purpose**: Receive constant data into registers

**Specializations**:
- `RECV_MIN`: Min quantization parameter
- `RECV_MAX`: Max quantization parameter
- `RECV_CFG`: Configuration register
- `RECV`: General constant data

**Example**:
```c
// config write
__builtin_IMCE_RECV_CFG(fifo_id);

// min write
__builtin_IMCE_RECV_MIN(fifo_id);

// max write
__builtin_IMCE_RECV_MAX(fifo_id);
```

#### RecvSendWrapper
**Purpose**: Wraps computation blocks with RECV/SEND operations

**Features**:
- Adds RECV for input edges (excluding constants)
- Adds SEND for output edges
- Deduplicates SEND when all outputs share same address
- Can create loops based on tensor shapes

**Example**:
```c
// Receive inputs
var_in0 = __builtin_IMCE_RECV(fifo_id);
var_in1 = __builtin_IMCE_RECV(fifo_id);
var_in2 = __builtin_IMCE_RECV(fifo_id);
var_in3 = __builtin_IMCE_RECV(fifo_id);

// Computation (from wrapped block)
var_out0 = __builtin_IMCE_ADD(var_in0, var_in1, 15);
var_out1 = __builtin_IMCE_ADD(var_in2, var_in3, 15);
// ...

// Send outputs
__builtin_IMCE_SEND(address, var_out0, fifo_id, 0);
__builtin_IMCE_SEND(address, var_out1, fifo_id, 0);
__builtin_IMCE_SEND(address, var_out2, fifo_id, 0);
__builtin_IMCE_SEND(address, var_out3, fifo_id, 0);
```

---

## Complete Example: Fused Convolution + ReLU

**Relay Operation**: `composite_function(qconv -> relu)`

**Generated Code**:

### INIT Phase:
```c
// config write
__builtin_IMCE_RECV_CFG(fifo_id);
```

### EXEC Phase:
```c
// generate: conv exec0
// outer_loop(iterate row offset) - row_group0
for (int i0 = 0; i0 < 8; i0++) {
  // inner_loop(iterate col offset. load inputs) - col_group0
  for (int i1 = 0; i1 < 8; i1++) {
    // load_block
    for (int i2 = 0; i2 < 6; i2++) {
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
      __builtin_IMCE_LOAD_LB(0);
    }
    __builtin_IMCE_STEP();
    
    // Get convolution results
    var0 = __builtin_IMCE_GET_CREG((short)0);
    var1 = __builtin_IMCE_GET_CREG((short)1);
    var2 = __builtin_IMCE_GET_CREG((short)2);
    var3 = __builtin_IMCE_GET_CREG((short)3);
    
    // ReLU post-op
    var_relu0 = __builtin_IMCE_MAXI(var0, 0);
    var_relu1 = __builtin_IMCE_MAXI(var1, 0);
    var_relu2 = __builtin_IMCE_MAXI(var2, 0);
    var_relu3 = __builtin_IMCE_MAXI(var3, 0);
    
    // Send results
    __builtin_IMCE_SEND(addr, var_relu0, 2, 0);
    __builtin_IMCE_SEND(addr, var_relu1, 2, 0);
    __builtin_IMCE_SEND(addr, var_relu2, 2, 0);
    __builtin_IMCE_SEND(addr, var_relu3, 2, 0);
  }
}
// endgenerate: conv exec0
```

---

## Execution Phases

### INIT Phase
- Receives and stores constant data (weights, biases, configs)
- Executed once before main computation
- Uses special RECV instructions for different register types

### EXEC Phase
- Main computation loop
- Contains convolution and post-operation chains
- Manages data flow with RECV/SEND

---

## Key Design Patterns

### 1. Post-Operation Chaining
Operations can be chained as post-ops to convolution:
```
ConvBlock -> AddBlock -> ReLUBlock -> MinmaxQuantBlock
```

Each post-op:
- Reads from previous operation's output
- Produces its own output
- Can be part of the SEND chain

### 2. Bitplane Parallelism
Most operations process 4 bitplanes in parallel:
```c
var0 = operation(input0, ...);
var1 = operation(input1, ...);
var2 = operation(input2, ...);
var3 = operation(input3, ...);
```

### 3. RECV/SEND Management
- Standalone operations: Wrapped with RecvSendWrapper
- Post-ops: RECV/SEND handled by ConvBlock or wrapper
- Constant data: Received in INIT phase, used in EXEC

### 4. Loop Structure for Convolution
```
Row Groups (outer)
  -> Column Groups (middle)
    -> Load Cycles (inner)
      -> Bitplane Operations (unrolled)
```

---

## IMCE Intrinsics Reference

### Data Movement
- `__builtin_IMCE_RECV(fifo_id)`: Receive data from FIFO
- `__builtin_IMCE_SEND(addr, data, fifo_id, flags)`: Send data to address
- `__builtin_IMCE_LOAD_LB(fifo_id)`: Load data into line buffer
- `__builtin_IMCE_RECV_MIN(fifo_id)`: Receive min quantization parameter
- `__builtin_IMCE_RECV_MAX(fifo_id)`: Receive max quantization parameter
- `__builtin_IMCE_RECV_CFG(fifo_id)`: Receive configuration

### Computation
- `__builtin_IMCE_ADD(a, b, mask)`: Vector addition
- `__builtin_IMCE_SUB(a, b, mask)`: Vector subtraction
- `__builtin_IMCE_MULTL(a, b, mask)`: Vector multiply (low)
- `__builtin_IMCE_MULTH(a, b, mask)`: Vector multiply (high)
- `__builtin_IMCE_DIV(a, b, mask)`: Vector division
- `__builtin_IMCE_MAXI(a, imm)`: Vector max with immediate (ReLU)
- `__builtin_IMCE_MINI(a, imm)`: Vector min with immediate
- `__builtin_IMCE_OR(a, b, mask)`: Vector bitwise OR (concat)

### Convolution
- `__builtin_IMCE_STEP()`: Execute convolution step
- `__builtin_IMCE_GET_CREG(idx)`: Get convolution result register

### Quantization
- `__builtin_IMCE_MM_QUANT(data, rs2, mask, qreg_idx)`: Min-max quantization
- `__builtin_IMCE_GET_QREG(idx)`: Get quantization result register

---

## Summary

The IMCE code generation system uses a two-layer architecture:

1. **Handlers** detect Relay operations and create appropriate code blocks
2. **Code Blocks** generate actual C++ intrinsic calls

This separation allows:
- Clean abstraction between graph-level and instruction-level concerns
- Flexible composition of operations (post-op chaining)
- Efficient bitplane-parallel code generation
- Support for both fused and standalone operations
