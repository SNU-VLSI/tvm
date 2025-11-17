# TVM External Compiler Constant 처리 메커니즘 분석

## 핵심 개념

**중요**: External compiler가 생성한 함수는 **두 가지 형태**로 호출됩니다:
1. **실행 함수**: `func_name(input0, input1, ..., output0, output1, ...)`
2. **초기화 함수**: `__init_func_name(constants_array)` - constant 데이터를 전달

## 파일 및 클래스 구조

### 1. Constant 수집 및 등록 (Python 측)

#### `python/tvm/relay/backend/contrib/imcflow/ext_codegen.py`
**목적**: External compiler codegen, constant 수집 및 C 코드 생성

**핵심 함수**:
```python
@tvm._ffi.register_func("relay.ext.imcflow.constant_updater")
def imcflow_constant_updater(expr, symbol):
    """
    Relay function에서 constant 추출
    
    반환값: Map<String, NDArray>
      - Key: constant 변수 이름 (예: "imcflow_subgraph_0_const_0")
      - Value: NDArray 데이터
    """
    # 현재는 빈 dict 반환 - 이것이 문제!
    return dict()

@tvm._ffi.register_func("relay.ext.imcflow")
def imcflow_external_codegen(func: relay.Function):
    """
    C source module 생성
    
    반환값: CSourceModule
      - code: 생성된 C 코드
      - func_names: [func_name]
      - const_vars: [constant 이름들] - 이것이 핵심!
    """
    func_name = func.attrs["global_symbol"]
    code = makeKernelStartCode(func_name, func, DevConfig.HOST_OS)
    const_vars = []  # TODO: constant 이름들을 추가해야 함
    
    return tvm.runtime._ffi_api.CSourceModuleCreate(
        code, "cc", [func_name], const_vars
    )
```

### 2. Constant 관리 (C++ 측 - Backend)

#### `src/relay/backend/utils.h`
**핵심 구조체/함수**:

```cpp
struct ConstantUpdater : public ExprVisitor {
  /**
   * Relay function을 순회하며 ConstantNode를 찾아 params에 저장
   * 
   * 각 constant마다:
   *   - 이름: "{symbol}_const_{idx}"
   *   - 데이터: const->data (NDArray)
   */
  void VisitExpr_(const ConstantNode* cn) final {
    std::string name = symbol_ + "_const_" + std::to_string(const_idx_++);
    (*params_)[name] = cn->data;
  }
};

void UpdateConstants(BaseFunc func, 
                    std::unordered_map<std::string, runtime::NDArray>* params) {
  /**
   * External function의 constant를 수집하는 통합 함수
   * 
   * 1. "relay.ext.{compiler}.constant_updater" 콜백 호출
   * 2. 없으면 기본 ConstantUpdater 사용
   * 3. params에 constant 이름 -> NDArray 매핑 저장
   */
}
```

#### `src/relay/backend/build_module.cc`
**라인 480-500**: External module의 constant 제거 로직

```cpp
auto ext_mods = executor_codegen_->GetExternalModules();
ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, ...);

// External module에 의해 소유된 constant를 전역 params에서 제거
for (tvm::runtime::Module mod : ext_mods) {
  auto pf_var = mod.GetFunction("get_const_vars");
  if (pf_var != nullptr) {
    Array<String> variables = pf_var();
    for (size_t i = 0; i < variables.size(); i++) {
      // ret_.params에서 해당 constant 제거
      ret_.params.erase(variables[i]);
    }
  }
}
```

### 3. Module 래핑 및 초기화

#### `src/target/source/source_module.cc`
**클래스**: `CSourceModuleNode`

```cpp
class CSourceModuleNode : public runtime::ModuleNode {
  Array<String> const_vars_;  // Constant 변수 이름 목록
  Array<String> func_names_;  // 함수 이름 목록
  
  PackedFunc GetFunction(const String& name, ...) {
    if (name == "get_const_vars") {
      // Constant 변수 이름 목록 반환
      return PackedFunc([this](...) { *rv = this->const_vars_; });
    }
    // ... 실제 함수 실행
  }
};
```

#### `src/target/metadata_module.cc`
**함수**: `CreateMetadataModule`, `CreateCppMetadataModule`

**라인 190-253**:
```cpp
runtime::Module CreateMetadataModule(...) {
  // External module을 두 그룹으로 분류:
  // 1. CRT-exportable (const_vars가 비어있음)
  // 2. Non-CRT-exportable (const_vars가 있음 - 초기화 필요)
  
  for (tvm::runtime::Module mod : ext_mods) {
    auto pf_sym = mod.GetFunction("get_symbol");
    auto pf_var = mod.GetFunction("get_const_vars");
    
    if (pf_sym != nullptr && pf_var != nullptr) {
      String symbol = pf_sym();
      Array<String> variables = pf_var();
      const_vars_by_symbol[symbol] = variables;  // symbol -> [const names]
    }
  }
}

static runtime::Module CreateCppMetadataModule(...) {
  // ConstLoaderModule 생성!
  runtime::Module const_loader_mod =
      runtime::ConstLoaderModuleCreate(const_var_ndarray, const_vars_by_symbol);
  
  const_loader_mod.Import(target_module);
  for (const auto& it : non_crt_exportable_modules) {
    const_loader_mod.Import(it);
  }
}
```

### 4. Runtime Constant 로딩

#### `src/runtime/const_loader_module.cc`
**클래스**: `ConstLoaderModuleNode` - **핵심!**

```cpp
class ConstLoaderModuleNode : public ModuleNode {
  // symbol -> [const var names] 매핑
  std::unordered_map<std::string, std::vector<std::string>> const_vars_by_symbol_;
  
  // const var name -> NDArray 매핑
  std::unordered_map<std::string, NDArray> const_var_ndarray_;
  
  // symbol -> 초기화 여부 플래그
  std::unordered_map<std::string, bool> initialized_;
  
  PackedFunc GetFunction(const String& name, ...) {
    // 첫 번째 호출 시 초기화
    if (initialized_.count(name) && !initialized_.at(name)) {
      this->InitSubModule(name);  // ← 여기서 __init_ 호출!
      initialized_[name] = true;
    }
    
    // 실제 함수는 imported module에서 찾음
    for (Module it : this->imports()) {
      PackedFunc pf = it.GetFunction(name);
      if (pf != nullptr) return pf;
    }
  }
  
  void InitSubModule(const std::string& symbol) {
    /**
     * 핵심 초기화 로직!
     * 
     * 1. "__init_{symbol}" 함수를 imported module에서 찾음
     * 2. 해당 symbol에 필요한 constant NDArray들을 가져옴
     * 3. init(constants_array) 호출
     */
    for (Module it : this->imports()) {
      std::string init_name = "__init_" + symbol;
      PackedFunc init = it.GetFunction(init_name, false);
      
      if (init != nullptr) {
        auto md = GetRequiredConstants(symbol);  // Array<NDArray>
        int ret = init(md);  // ← Constant 전달!
        break;
      }
    }
  }
  
  Array<NDArray> GetRequiredConstants(const std::string& symbol) {
    /**
     * symbol에 필요한 constant들을 순서대로 반환
     * 
     * const_vars_by_symbol_[symbol] = ["const_0", "const_1", ...]
     * → [const_var_ndarray_["const_0"], const_var_ndarray_["const_1"], ...]
     */
    Array<NDArray> ret;
    std::vector<std::string> vars = const_vars_by_symbol_[symbol];
    for (const auto& var : vars) {
      ret.push_back(const_var_ndarray_[var]);
    }
    return ret;
  }
};
```

### 5. JSON Runtime의 Constant 처리 (참고용)

#### `src/runtime/contrib/json/json_runtime.h`
**클래스**: `JSONRuntimeBase`

```cpp
class JSONRuntimeBase : public ModuleNode {
  Array<String> const_names_;  // 필요한 constant 이름들
  bool initialized_;           // 초기화 여부
  
  PackedFunc GetFunction(const String& name, ...) {
    if ("__init_" + this->symbol_name_ == name) {
      // 초기화 함수: constant array를 받아 저장
      return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1U);
        if (!this->initialized_) {
          this->Init(args[0]);  // Array<NDArray> 전달
          this->initialized_ = true;
        }
        *rv = 0;
      });
    } else if (this->symbol_name_ == name) {
      // 실행 함수
      return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_);
        this->SetInputOutputBuffers(args);
        this->Run();
      });
    } else if (name == "get_const_vars") {
      return PackedFunc([this](...) { *rv = this->const_names_; });
    }
  }
};
```

### 6. Graph Executor에서 함수 호출

#### `src/runtime/graph_executor/graph_executor.cc`

**`SetupOpExecs()` - 라인 504-580**:
```cpp
void GraphExecutor::SetupOpExecs() {
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    const auto& inode = nodes_[nid];
    
    // Args 구성: inputs + outputs
    std::vector<DLTensor*> args;
    
    // 1. Input 텐서들 추가
    for (const auto& e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      args.push_back(data_entry_[eid].operator->());
    }
    
    // 2. Output 텐서들 추가
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(data_entry_[eid].operator->());
    }
    
    // 3. CreateTVMOp: args를 TVMArgs로 변환하여 함수 호출 준비
    op_execs_[nid] = CreateTVMOp(inode.param, args);
  }
}
```

**`CreateTVMOp()` - 라인 585-635**:
```cpp
std::pair<std::function<void()>, std::shared_ptr<OpArgs>>
GraphExecutor::CreateTVMOp(const TVMOpParam& param, 
                          const std::vector<DLTensor*>& args) {
  // DLTensor* 벡터를 TVMValue 배열로 변환
  for (size_t i = 0; i < args.size(); ++i) {
    TVMValue v;
    v.v_handle = args[i];
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kTVMDLTensorHandle);
  }
  
  // Module에서 함수 가져오기
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, true);
  
  // 실행 함수 생성
  auto fexec = [arg_ptr, pf]() {
    TVMArgs targs(arg_ptr->arg_values.data(), 
                  arg_ptr->arg_tcodes.data(),
                  arg_ptr->arg_values.size());
    pf.CallPacked(targs, &rv);  // ← 여기서 실제 함수 호출!
  };
  
  return {fexec, arg_ptr};
}
```

**`LoadParams()` - 라인 315-325**:
```cpp
void GraphExecutor::LoadParams(dmlc::Stream* strm) {
  /**
   * Param 파일에서 constant를 로드하여 input으로 설정
   * 
   * 주의: 이것은 graph의 input constant를 위한 것
   * External module의 constant는 ConstLoaderModule이 처리!
   */
  Map<String, NDArray> params = ::tvm::runtime::LoadParams(strm);
  for (auto& p : params) {
    int in_idx = GetInputIndex(p.first);
    if (in_idx < 0) continue;
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    data_entry_[eid].CopyFrom(p.second);
  }
}
```

## 완전한 Constant 처리 흐름

### 컴파일 타임 (Build Time)

```
1. Relay IR → Partitioning → External Functions

2. 각 External Function에 대해:
   a. constant_updater(func, symbol) 호출
      → Map<String, NDArray> 반환
      예: {"imcflow_subgraph_0_const_0": weight_array,
           "imcflow_subgraph_0_const_1": bias_array}
   
   b. external_codegen(func) 호출
      → CSourceModule 생성
         - code: C 코드
         - func_names: ["imcflow_subgraph_0"]
         - const_vars: ["imcflow_subgraph_0_const_0", 
                        "imcflow_subgraph_0_const_1"]

3. CreateMetadataModule():
   a. const_vars_by_symbol 구성:
      {
        "imcflow_subgraph_0": ["imcflow_subgraph_0_const_0",
                               "imcflow_subgraph_0_const_1"]
      }
   
   b. const_var_ndarray 구성:
      {
        "imcflow_subgraph_0_const_0": weight_array,
        "imcflow_subgraph_0_const_1": bias_array
      }
   
   c. ConstLoaderModule 생성:
      ConstLoaderModule
        ├─ imports: [CSourceModule, ...]
        ├─ const_vars_by_symbol_
        └─ const_var_ndarray_

4. MLF (Model Library Format) 저장:
   - graph.json: 그래프 구조
   - params: 일반 params (external constant 제외)
   - module: ConstLoaderModule + CSourceModule
```

### 런타임 (Runtime)

```
1. GraphExecutor 초기화:
   graph_executor = GraphExecutor(module, device)
   graph_executor.load_params(params)  # 일반 params만

2. 첫 실행 시 (graph_executor.run()):
   a. SetupOpExecs()에서 각 op에 대해 CreateTVMOp()
   
   b. External function "imcflow_subgraph_0" 실행 전:
      
      i. module_.GetFunction("imcflow_subgraph_0") 호출
         → ConstLoaderModule::GetFunction()
      
      ii. initialized_["imcflow_subgraph_0"] == false 확인
          → InitSubModule("imcflow_subgraph_0") 호출
      
      iii. InitSubModule():
           - "__init_imcflow_subgraph_0" 함수 찾기
           - GetRequiredConstants("imcflow_subgraph_0") 호출
             → [weight_array, bias_array] 반환
           - init_func([weight_array, bias_array]) 호출
      
      iv. CSourceModule의 "__init_imcflow_subgraph_0":
          - Array<NDArray>를 받아서 저장
          - 이후 실행 시 사용
      
      v. initialized_["imcflow_subgraph_0"] = true

3. 실제 함수 호출:
   a. ConstLoaderModule::GetFunction("imcflow_subgraph_0")
      → CSourceModule::GetFunction("imcflow_subgraph_0")
   
   b. PackedFunc 호출:
      args = [input0_handle, input1_handle, output0_handle]
      func(args)  # Input/Output만 전달!
   
   c. C 함수 내부에서:
      - 이미 저장된 constant 사용
      - 또는 constant를 함수 내부 static 변수로 관리
```

## 왜 Constant는 Input/Output Args에 포함되지 않는가?

### 이유 1: 초기화 단계 분리
- **초기화**: `__init_` 함수로 한 번만 전달 (무거운 데이터)
- **실행**: Input/Output만 전달 (가벼운 인터페이스)

### 이유 2: 메모리 효율성
```cpp
// External module 내부 구현 예시:
static NDArray* cached_weights = nullptr;
static NDArray* cached_bias = nullptr;

int32_t __init_my_conv(TVMArgs args, ...) {
  // 한 번만 실행
  cached_weights = args[0];
  cached_bias = args[1];
  return 0;
}

int32_t my_conv(TVMArgs args, ...) {
  DLTensor* input = args[0];
  DLTensor* output = args[1];
  
  // Cached constant 사용
  conv2d(input->data, cached_weights->data, 
         cached_bias->data, output->data);
  return 0;
}
```

### 이유 3: 인터페이스 일관성
- 모든 TVM operator: `func(inputs..., outputs...)`
- External function도 동일한 인터페이스 유지
- Constant는 별도 초기화 메커니즘으로 처리

### 이유 4: Graph Executor 구조
```cpp
// Graph executor는 node의 inputs/outputs만 알고 있음
for (const auto& e : inode.inputs) {
  args.push_back(data_entry_[e]);  // Graph edge만
}
for (uint32_t i = 0; i < inode.num_outputs; ++i) {
  args.push_back(data_entry_[output_eid]);
}
// Constant는 graph edge가 아니므로 여기에 없음!
```

## 핵심 포인트 요약

1. **두 함수 패턴**:
   - `__init_{symbol}(Array<NDArray>)`: Constant 초기화
   - `{symbol}(inputs..., outputs...)`: 실제 실행

2. **const_vars의 역할**:
   - External module이 필요로 하는 constant 변수 이름 목록
   - ConstLoaderModule이 이를 사용하여 초기화

3. **ConstLoaderModule**:
   - Constant 저장소 및 초기화 관리자
   - 첫 호출 시 `__init_` 함수로 constant 전달
   - 이후 실제 함수를 imported module로 위임

4. **순서 중요**:
   - constant_updater와 external_codegen에서 constant 순서 일치 필수
   - 같은 visitor 로직 사용 권장

5. **현재 imcflow의 문제**:
   - `constant_updater`: 빈 dict 반환
   - `external_codegen`: const_vars 비어있음
   - → Constant가 전혀 전달되지 않음!

