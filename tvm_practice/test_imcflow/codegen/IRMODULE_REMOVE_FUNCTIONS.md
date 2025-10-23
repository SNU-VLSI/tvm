# IRModule에서 GlobalVar 제거 함수들

## 발견된 함수

### 1. `IRModule.__delitem__()` ✅ 추천

**위치**: `python/tvm/ir/module.py:145`

**시그니처**:
```python
def __delitem__(self, var: Union[str, _expr.GlobalVar]):
    _ffi_api.Module_Remove(self, var)
```

**사용법**:
```python
# 방법 1: String으로 삭제
del mod["function_name"]

# 방법 2: GlobalVar로 삭제
gv = relay.GlobalVar("function_name")
del mod[gv]
```

**특징**:
- Python의 표준 `del` 연산자 사용
- String 또는 GlobalVar 모두 지원
- 가장 Pythonic한 방법

### 2. `_ffi_api.Module_Remove()` (Low-level)

**위치**: C++ FFI - `src/ir/module.cc:436`

**C++ 구현**:
```cpp
TVM_REGISTER_GLOBAL("ir.Module_Remove")
    .set_body_typed([](IRModule mod, Variant<String, GlobalVar> var) -> IRModule {
      GlobalVar gvar = [&]() {
        if (auto opt = var.as<GlobalVar>()) {
          return opt.value();
        } else if (auto opt = var.as<String>()) {
          return mod->GetGlobalVar(opt.value());  // String을 GlobalVar로 변환
        }
      }();
      mod->Remove(gvar);  // 실제 제거
      return mod;
    });
```

**사용법** (직접 사용 권장하지 않음):
```python
from tvm.ir import _ffi_api

_ffi_api.Module_Remove(mod, "function_name")
# 또는
_ffi_api.Module_Remove(mod, global_var)
```

**특징**:
- Low-level FFI 함수
- 내부적으로 `__delitem__`이 이것을 호출
- 직접 사용할 이유 없음 (대신 `del mod[...]` 사용)

## 관련 함수들

### `IRModule.__contains__()` - 존재 확인

```python
def __contains__(self, var: Union[str, _expr.GlobalVar]) -> bool:
    return _ffi_api.Module_Contains(self, var)
```

**사용법**:
```python
if "my_func" in mod:
    del mod["my_func"]

# 또는
gv = relay.GlobalVar("my_func")
if gv in mod:
    del mod[gv]
```

### `IRModule.get_global_var()` - GlobalVar 가져오기

```python
def get_global_var(self, name: str) -> GlobalVar:
    ...
```

**사용법**:
```python
gv = mod.get_global_var("function_name")
del mod[gv]
```

### `IRModule.get_global_vars()` - 모든 GlobalVar 가져오기

```python
def get_global_vars(self) -> List[GlobalVar]:
    ...
```

**사용법**:
```python
# 모든 GlobalVar 순회
for gv in mod.get_global_vars():
    print(gv.name_hint)
    if should_remove(gv):
        del mod[gv]
```

## 사용 예제

### 예제 1: 단순 삭제

```python
import tvm
from tvm import relay

mod = tvm.IRModule()
x = relay.var("x", shape=(10,), dtype="float32")
func = relay.Function([x], relay.nn.relu(x))
mod["my_func"] = func

# 삭제
del mod["my_func"]
```

### 예제 2: 조건부 삭제 (안전한 패턴)

```python
# ✓ 올바른 방법: 먼저 수집, 나중에 삭제
to_remove = []
for gv in mod.get_global_vars():
    if gv.name_hint.startswith("_temp_"):
        to_remove.append(gv.name_hint)

for name in to_remove:
    del mod[name]
```

### 예제 3: 존재 확인 후 삭제

```python
# 방법 1: __contains__ 사용
if "my_func" in mod:
    del mod["my_func"]

# 방법 2: try-except
try:
    del mod["my_func"]
except ValueError:
    print("Function not found")

# 방법 3: 헬퍼 함수
def safe_remove(mod, name):
    try:
        del mod[name]
        return True
    except ValueError:
        return False

if safe_remove(mod, "my_func"):
    print("Deleted")
```

### 예제 4: 필터링

```python
# imcflow 함수만 남기고 나머지 삭제
to_remove = []
for gv in mod.get_global_vars():
    func = mod[gv]
    if "Compiler" not in func.attrs or func.attrs["Compiler"] != "imcflow":
        to_remove.append(gv.name_hint)

for name in to_remove:
    del mod[name]
```

## 비교: 제거 vs 새 모듈 생성

### 방법 1: In-place 제거 (del)
```python
for name in to_remove:
    del mod[name]
```
- **장점**: 메모리 효율적, 빠름
- **단점**: 원본 수정됨

### 방법 2: 새 모듈 생성
```python
new_mod = tvm.IRModule()
for gv in old_mod.get_global_vars():
    if should_keep(gv):
        new_mod[gv] = old_mod[gv]
```
- **장점**: 원본 보존
- **단점**: 메모리 사용 증가

## 요약

| 함수 | 용도 | 추천도 |
|------|------|--------|
| `del mod["name"]` | GlobalVar 제거 | ⭐⭐⭐⭐⭐ 추천 |
| `del mod[gv]` | GlobalVar 제거 | ⭐⭐⭐⭐⭐ 추천 |
| `_ffi_api.Module_Remove()` | Low-level 제거 | ⭐ 직접 사용 X |
| `"name" in mod` | 존재 확인 | ⭐⭐⭐⭐⭐ 유용 |
| `mod.get_global_var()` | GlobalVar 가져오기 | ⭐⭐⭐⭐ 유용 |
| `mod.get_global_vars()` | 모든 GlobalVar | ⭐⭐⭐⭐⭐ 필수 |

## 핵심 포인트

1. **`del mod["name"]` 또는 `del mod[gv]` 사용** - 가장 Pythonic
2. **존재하지 않으면 ValueError** - 안전하게 처리 필요
3. **Iteration 중 삭제 주의** - 먼저 수집 후 삭제
4. **String과 GlobalVar 모두 지원** - 편한 방법 선택
5. **In-place 수정** - 원본이 변경됨

## 결론

**IRModule에서 GlobalVar를 제거하는 표준 방법:**

```python
# 이것만 기억하세요!
del mod["function_name"]
```

간단하고 명확합니다! 🎯
