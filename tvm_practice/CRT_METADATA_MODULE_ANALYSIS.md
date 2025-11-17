# CreateCrtMetadataModule 에러 분석 및 TVM 설계 논리

## 문제 상황

`CSourceModuleCreate`에서 `const_vars`를 전달하면 `CreateCrtMetadataModule`에서 에러 발생:
```
CHECK(false) << "These X modules are not exportable to C-runtime: module_name"
```

## TVM의 설계 논리

### 1. CRT (C Runtime) vs C++ Runtime의 차이

TVM은 두 가지 런타임을 지원합니다:

#### **C++ Runtime** (`runtime->name == "crt"` **아닌** 경우)
- **특징**: 동적 초기화 지원
- **Constant 처리**: `ConstLoaderModule` 사용
- **메커니즘**: `__init_` 함수로 런타임에 constant 전달
- **사용 환경**: 일반 Linux/서버 환경

#### **CRT (C Runtime)** (`runtime->name == "crt"`)
- **특징**: 완전히 정적인 C 환경
- **Constant 처리**: 컴파일 타임에 모든 데이터 임베드
- **메커니즘**: Constant를 C 배열로 코드에 직접 포함
- **사용 환경**: 임베디드, 마이크로컨트롤러, 베어메탈

### 2. Module 분류 로직 (metadata_module.cc:210-240)

```cpp
// External module을 두 그룹으로 분류:
for (tvm::runtime::Module mod : ext_modules) {
  auto pf_sym = mod.GetFunction("get_symbol");
  auto pf_var = mod.GetFunction("get_const_vars");
  
  std::vector<std::string> symbol_const_vars;
  if (pf_sym != nullptr && pf_var != nullptr) {
    Array<String> variables = pf_var();
    for (auto var : variables) {
      symbol_const_vars.push_back(var);
    }
  }
  
  // ★ 핵심 분류 로직 ★
  if (symbol_const_vars.empty() &&           // 1. const_vars가 비어있고
      is_targeting_crt &&                     // 2. CRT를 타겟으로 하고
      mod->IsDSOExportable() &&               // 3. DSO exportable이고
      (target->kind->name == "c" || 
       target->kind->name == "llvm")) {       // 4. C 또는 LLVM 타겟이면
    
    crt_exportable_modules.push_back(mod);    // → CRT exportable!
    
  } else {
    non_crt_exportable_modules.push_back(mod); // → Non-CRT exportable
  }
}
```

### 3. CRT의 제약사항 (metadata_module.cc:48-65)

```cpp
static runtime::Module CreateCrtMetadataModule(...) {
  // ★ CRT는 non-exportable module을 허용하지 않음! ★
  if (!non_crt_exportable_modules.empty()) {
    std::string non_exportable_modules;
    for (auto mod : non_crt_exportable_modules) {
      auto pf_sym = mod.GetFunction("get_symbol");
      if (pf_sym != nullptr) {
        non_exportable_modules += pf_sym().operator std::string();
      }
    }
    
    // ← 여기서 에러 발생!
    CHECK(false) << "These " << non_crt_exportable_modules.size()
                 << " modules are not exportable to C-runtime: " 
                 << non_exportable_modules;
  }
  // ...
}
```

## 왜 이런 설계인가?

### 이유 1: CRT의 철학 - "완전한 정적 컴파일"

CRT는 임베디드 시스템을 위한 것으로, 다음을 요구합니다:
- **No dynamic loading**: 모든 것이 컴파일 타임에 결정
- **No runtime initialization**: `__init_` 같은 동적 초기화 불가
- **No heap allocation**: 모든 메모리가 정적 할당
- **No C++ features**: Pure C만 사용

### 이유 2: Constant의 두 가지 처리 방식

#### **방식 1: 동적 초기화** (C++ Runtime, non-exportable)
```cpp
// ConstLoaderModule 생성
// runtime에 __init_ 함수 호출하여 constant 전달

// 장점: 유연함, 대용량 constant 가능
// 단점: 동적 메모리, 초기화 오버헤드
```

#### **방식 2: 정적 임베딩** (CRT, exportable)
```c
// 컴파일 타임에 constant를 C 배열로 생성
static const float weights[] = {1.0, 2.0, 3.0, ...};

// 장점: 완전 정적, 빠른 시작
// 단점: 코드 크기 증가, 컴파일 타임 증가
```

### 이유 3: `const_vars`의 의미

```python
# const_vars가 비어있음 = "이 모듈은 constant가 없습니다"
CSourceModuleCreate(code, "c", [func_name], [])  # ✓ CRT OK

# const_vars가 있음 = "이 모듈은 런타임 초기화가 필요합니다"
CSourceModuleCreate(code, "c", [func_name], ["const_0"])  # ✗ CRT 불가!
```

## 문제의 근본 원인

당신의 코드:
```python
@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  const_vars = list(DevConfig().ImcflowFuncMap[func_name].const_name_map.values())
  
  # const_vars가 비어있지 않음!
  # → non_crt_exportable_modules로 분류됨
  # → CRT 타겟팅 시 에러 발생
  return tvm.runtime._ffi_api.CSourceModuleCreate(
      code, "c", [String(func_name)], const_vars)
```

**결과**:
1. `const_vars`가 있으므로 → `non_crt_exportable_modules`에 추가됨
2. CRT 타겟팅 중 → `CreateCrtMetadataModule` 호출
3. `non_crt_exportable_modules`가 비어있지 않음 → **CHECK 실패!**

## 해결 방법

### 옵션 1: C++ Runtime 사용 (권장)

CRT 대신 C++ Runtime을 사용하면 문제 해결:

```python
# Build script에서:
executor = relay.build_module.Executor("graph", {"link-params": True})
runtime = relay.build_module.Runtime("cpp", {"system-lib": True})  # CRT 대신 cpp

# 이렇게 하면:
# - is_targeting_crt = False
# - CreateCppMetadataModule 호출
# - ConstLoaderModule 사용하여 constant 동적 초기화
```

### 옵션 2: Constant를 C 코드에 직접 임베드 (CRT 방식)

CRT를 계속 사용하려면, constant를 코드에 직접 포함:

```python
@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
  # 1. Constant 데이터를 수집
  constants = collect_constants(func)
  
  # 2. C 코드에 constant 배열 생성
  const_arrays_code = generate_constant_arrays(constants)
  
  # 3. 코드에 포함
  code = const_arrays_code + makeKernelStartCode(func_name, func, DevConfig.HOST_OS)
  
  # 4. const_vars는 비워둠! (CRT exportable)
  return tvm.runtime._ffi_api.CSourceModuleCreate(
      code, "c", [String(func_name)], [])  # ← 빈 배열!
```

예시:
```c
// 생성되는 코드:
static const float imcflow_weight_0[] = {
  1.0f, 2.0f, 3.0f, // ... 모든 weight 데이터
};

static const float imcflow_bias_0[] = {
  0.1f, 0.2f, // ... 모든 bias 데이터
};

// Kernel 함수에서 직접 사용
void my_conv_kernel(float* input, float* output) {
  conv2d(input, imcflow_weight_0, imcflow_bias_0, output);
}
```

### 옵션 3: Hybrid 접근 (실용적)

Imcflow의 경우, NPU로 데이터를 전송하므로:

```python
def imcflow_external_codegen(func: relay.Function):
  # Constant는 NPU memory에 있으므로
  # Host C 코드에서는 constant 접근 불필요
  
  # 1. Constant 데이터를 binary로 생성 (object file)
  # 2. Kernel 함수는 input/output만 처리
  # 3. const_vars는 비움
  
  code = makeKernelStartCode(func_name, func, DevConfig.HOST_OS)
  
  # const_vars 비움 → CRT exportable
  return tvm.runtime._ffi_api.CSourceModuleCreate(
      code, "c", [String(func_name)], [])
```

## TVM의 설계 의도 요약

### "CRT = 완전한 정적 컴파일, 동적 요소 없음"

```
┌─────────────────────────────────────────────────────────┐
│                    TVM Runtime 선택                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  C++ Runtime                    CRT                      │
│  ├─ 동적 초기화 OK              ├─ 완전 정적              │
│  ├─ ConstLoaderModule           ├─ 모든 constant 임베드   │
│  ├─ __init_ 함수                ├─ __init_ 불가          │
│  ├─ const_vars 사용 가능        ├─ const_vars 불가       │
│  ├─ Heap 메모리 사용            ├─ Stack/Static만        │
│  └─ 일반 Linux/서버             └─ 임베디드/베어메탈       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 분류 기준

```
External Module
    │
    ├─ const_vars가 있는가?
    │   │
    │   ├─ YES → non_crt_exportable
    │   │         └─ C++ Runtime 필요 (ConstLoaderModule)
    │   │
    │   └─ NO → IsDSOExportable?
    │             │
    │             ├─ YES → crt_exportable
    │             │         └─ CRT 가능
    │             │
    │             └─ NO → non_crt_exportable
    │                       └─ C++ Runtime 필요
```

## 당신의 상황 분석

### 현재 문제:
```python
# constant_updater: params 반환 (constant 있음)
# external_codegen: const_vars 추가

# → Module이 non_crt_exportable로 분류됨
# → CRT 빌드 시 에러 발생
```

### 해결책 선택 가이드:

#### **시나리오 1: NPU가 constant를 관리**
```python
# Constant는 NPU 메모리에 있고
# Host C 코드는 접근 불필요
# → const_vars를 비워도 됨

@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):
    # NPU용 constant는 별도 처리
    return {}  # 빈 dict 반환!

@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
    code = makeKernelStartCode(...)
    return CSourceModuleCreate(code, "c", [func_name], [])  # 빈 배열
```

#### **시나리오 2: Host가 constant 접근 필요**
```python
# C 코드에서 constant 데이터 필요
# → C++ Runtime 사용하거나 코드에 임베드

# 옵션 A: C++ Runtime
runtime = Runtime("cpp", {"system-lib": True})

# 옵션 B: Constant를 C 배열로 생성
code = generate_static_constants() + makeKernelStartCode(...)
return CSourceModuleCreate(code, "c", [func_name], [])
```

## 핵심 정리

1. **CRT = 모든 것이 정적, const_vars 불가**
2. **const_vars가 있으면 = 동적 초기화 필요 = C++ Runtime 필요**
3. **당신의 에러 = CRT 사용 + const_vars 제공 = 모순**
4. **해결 = (1) C++ Runtime 사용 또는 (2) const_vars 제거**

TVM의 설계는 합리적입니다:
- 임베디드 = CRT = 정적 = const_vars 없음
- 서버/일반 = C++ Runtime = 동적 = const_vars 가능

당신의 use case에 맞게 선택하세요!

