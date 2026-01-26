# Multicast Deadlock Issue with Multiple Post-ops

## Problem Summary

Conv block에 post_op가 2개 이상 있고, 그 중 하나가 외부 입력(예: residual add)을 받을 때, multicast receiver들의 진행 속도 차이로 인해 deadlock이 발생할 수 있다.

## Affected Model

- `resnet8_subset10_pretrained_super_small_evl` (실패)
- `resnet8_subset09_pretrained_super_small_evl` (성공 - post_op가 1개만 있음)

## Data Flow Diagram

```
                          ┌─────────────────────────────────────────────────┐
                          │                                                 │
                          ▼                                                 │
inode_0_0 ──(multicast, uuid=1)──┬──► imce_3_1 (multiply) ──(uuid=5)──┐    │
                                 │                                     │    │
                                 │                                     ▼    │
                                 └──► imce_3_4 ──► imce_3_3 ──► imce_3_2 ──► imce_2_4
                                      (quant)     (conv+bn)    (quant)      (conv+bn+add)
                                                                  │              ▲
                                                                  └──(uuid=4)────┘
```

## Key Components

### inode_0_0 (Sender - Multicast)
```cpp
for (int i1 = 0; i1 < var6; i1++) {
    __builtin_INODE_SEND(var4 + i1*32, 0, 1, 2);
    __builtin_INODE_STANDBY(16, 1);  // wait for imce_3_1
    __builtin_INODE_STANDBY(19, 1);  // wait for imce_3_4
    __builtin_INODE_SET_FLAG(1);
    __builtin_INODE_STANDBY(16, 0);
    __builtin_INODE_STANDBY(19, 0);
    __builtin_INODE_SET_FLAG(0);
}
```

### imce_3_1 (Multiply - Fast Path)
```cpp
for (int i1 = 0; i1 < 16; i1++) {
    // RECV from inode_0_0
    var31 = __builtin_IMCE_RECV(2);
    __builtin_IMCE_SETFLAG(1);
    __builtin_IMCE_STANDBY(0, 1);  // wait for sender
    __builtin_IMCE_SETFLAG(0);

    // Fast computation
    var32 = __builtin_IMCE_MULTL(var31, var30, 15);

    // SEND to imce_2_4 - BLOCKS until imce_2_4 is ready!
    __builtin_IMCE_SEND(1, var32, 2, 0);
    __builtin_IMCE_STANDBY(14, 5);  // <<< BLOCKING POINT
    __builtin_IMCE_SETFLAG(5);
    __builtin_IMCE_STANDBY(14, 0);
    __builtin_IMCE_SETFLAG(0);
}
```

### imce_2_4 (Conv + BatchNorm + Add)
```cpp
// 1. Conv input LOAD_LB (from imce_3_2, uuid=4)
for (int i1 = 0; i1 < 4; i1++) {
    for (int i2 = 0; i2 < 4; i2++) {
        __builtin_IMCE_LOAD_LB(0);
        __builtin_IMCE_SETFLAG(4);
        __builtin_IMCE_STANDBY(17, 4);  // wait for imce_3_2
        __builtin_IMCE_SETFLAG(0);
    }
}

// 2. Conv computation
__builtin_IMCE_STEP();
var5-8 = __builtin_IMCE_GET_CREG(...);

// 3. BatchNorm (post_op 1)
var20 = __builtin_IMCE_MULTL(var5, var14, 15);
var20 = __builtin_IMCE_ADD(var20, var15, 15);
// ...

// 4. RECV for Add rhs (post_op 2) - Only NOW ready to receive from imce_3_1!
var9 = __builtin_IMCE_RECV(2);
__builtin_IMCE_SETFLAG(5);
__builtin_IMCE_STANDBY(16, 5);  // <<< Only now signals imce_3_1
__builtin_IMCE_SETFLAG(0);

// 5. Add computation
var16 = __builtin_IMCE_ADD(var20, var9, 15);
```

## Deadlock Sequence

```
Time │ inode_0_0          │ imce_3_1           │ imce_3_4           │ imce_3_2→imce_2_4
─────┼────────────────────┼────────────────────┼────────────────────┼──────────────────
  1  │ SEND #1            │                    │                    │
     │ STANDBY(16,1)      │ RECV               │ RECV               │
     │ STANDBY(19,1)      │ SETFLAG(1) ✓       │ SETFLAG(1) ✓       │
     │ SETFLAG(1)         │ STANDBY(0,1)→pass  │ STANDBY(0,1)→pass  │
     │ sync complete      │ SETFLAG(0)         │ SETFLAG(0)         │
─────┼────────────────────┼────────────────────┼────────────────────┼──────────────────
  2  │                    │ MULTL (fast)       │ Continue chain...  │ Processing...
     │                    │ SEND to imce_2_4   │                    │
     │                    │ STANDBY(14,5) 🔒   │                    │ (not ready yet)
     │                    │ BLOCKED!           │                    │
─────┼────────────────────┼────────────────────┼────────────────────┼──────────────────
  3  │ SEND #2            │ (stuck at uuid=5)  │ RECV #2            │
     │ STANDBY(16,1) 🔒   │ flag = 0           │ SETFLAG(1)         │
     │ BLOCKED!           │                    │ waiting...         │
─────┼────────────────────┼────────────────────┼────────────────────┼──────────────────
  4  │ (waiting for       │ (waiting for       │ (waiting for       │ (waiting for
     │  imce_3_1 flag=1)  │  imce_2_4 flag=5)  │  inode_0_0 data)   │  imce_3_2 data)
     │         │          │         │          │         │          │         │
     │         └──────────┴─────────┴──────────┴─────────┴──────────┴─────────┘
     │                              DEADLOCK CYCLE
```

## Root Cause

1. **Multicast Constraint**: inode_0_0의 multicast는 **모든 receiver (imce_3_1, imce_3_4)가 ready**해야만 다음 데이터를 보낼 수 있음

2. **Speed Mismatch**:
   - imce_3_1 (multiply): RECV 1개 → 빠른 연산 → SEND (blocking)
   - imce_3_4 → ... → imce_2_4 chain: 긴 pipeline, conv 계산 필요

3. **Blocking SEND**: imce_3_1의 SEND는 imce_2_4가 RECV할 준비가 될 때까지 block됨

4. **Circular Dependency**:
   - imce_3_1이 다음 RECV를 하려면 → 현재 SEND가 완료되어야 함
   - SEND가 완료되려면 → imce_2_4가 RECV해야 함
   - imce_2_4가 RECV하려면 → conv input이 와야 함 (imce_3_2에서)
   - imce_3_2가 데이터를 보내려면 → imce_3_4 chain이 진행되어야 함
   - imce_3_4가 진행하려면 → inode_0_0에서 데이터를 받아야 함
   - inode_0_0이 데이터를 보내려면 → **imce_3_1이 ready해야 함** (multicast)
   - **Deadlock!**

## Solution Directions

### Option 1: Separate Multicast
Multiply의 input source를 conv chain의 input source와 분리하여 별도의 SEND-RECV pair로 만든다.

### Option 2: Buffered Producer
imce_3_1이 모든 input을 먼저 버퍼링한 후에 output을 보내도록 한다.
```
// Instead of: RECV → compute → SEND (repeat)
// Do: RECV all → compute all → SEND all
```

### Option 3: Async SEND without Blocking Sync
Post-op의 외부 입력 producer가 non-blocking으로 데이터를 보낼 수 있도록 sync 방식을 변경한다.

### Option 4: Reorder Operations
imce_2_4에서 Add의 RECV를 conv LOAD_LB와 병렬로 또는 이전에 수행하도록 재배치한다.

## Files Involved

- `/root/project/tvm/python/tvm/relay/backend/contrib/imcflow/imce_codeblock.py`: ConvBlock code generation
- `/root/project/tvm/python/tvm/relay/backend/contrib/imcflow/inode_codeblock.py`: INODE send/recv generation
- Generated files:
  - `resnet8_subset10_.../build/.../imce.cpp`
  - `resnet8_subset10_.../build/.../inode.cpp`
