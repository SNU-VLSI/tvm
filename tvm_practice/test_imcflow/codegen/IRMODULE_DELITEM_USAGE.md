# IRModule.__delitem__ 사용법

## 기본 사용법

`IRModule`에서 함수를 제거하는 방법입니다.

### 1. String으로 삭제

```python
mod = tvm.IRModule()
# ... add functions ...

del mod["function_name"]  # 함수 이름으로 삭제
```

### 2. GlobalVar로 삭제

```python
gv = relay.GlobalVar("function_name")
mod[gv] = some_function

del mod[gv]  # GlobalVar 객체로 삭제
```

## 특징

### ✅ 가능한 것들

1. **main 함수도 삭제 가능**
   ```python
   del mod["main"]  # 문제없음
   ```

2. **삭제 후 같은 이름으로 재추가**
   ```python
   del mod["my_func"]
   mod["my_func"] = new_function  # OK!
   ```

3. **대량 삭제**
   ```python
   for name in ["func1", "func2", "func3"]:
       del mod[name]
   ```

### ⚠️ 주의사항

1. **존재하지 않는 함수 삭제 시 에러 발생**
   ```python
   del mod["non_existent"]
   # ValueError: Cannot find global var "non_existent" in the Module
   ```
   
   **해결**: 먼저 존재 확인
   ```python
   if "my_func" in [gv.name_hint for gv in mod.get_global_vars()]:
       del mod["my_func"]
   ```

2. **삭제 중 iteration 주의**
   ```python
   # ✗ 잘못된 방법 (iteration 중 삭제)
   for gv in mod.get_global_vars():
       if condition:
           del mod[gv]  # 위험!
   
   # ✓ 올바른 방법 (먼저 수집 후 삭제)
   to_remove = []
   for gv in mod.get_global_vars():
       if condition:
           to_remove.append(gv.name_hint)
   
   for name in to_remove:
       del mod[name]
   ```

## 실용적 패턴

### 패턴 1: 조건부 필터링

```python
# 예: imcflow가 아닌 함수들 제거
to_remove = []
for gv in mod.get_global_vars():
    func = mod[gv]
    if "Compiler" not in func.attrs or func.attrs["Compiler"] != "imcflow":
        to_remove.append(gv.name_hint)

for name in to_remove:
    del mod[name]
```

결과:
```
Original module:
  func0: Compiler=imcflow
  func1: Compiler=none      <- 제거됨
  func2: Compiler=imcflow
  func3: Compiler=none      <- 제거됨
  func4: Compiler=imcflow

Filtered module (only imcflow):
  func0: Compiler=imcflow
  func2: Compiler=imcflow
  func4: Compiler=imcflow
```

### 패턴 2: 함수 교체

```python
# 1. 기존 함수 삭제
del mod["my_func"]

# 2. 새로운 함수 추가
x = relay.var("x", shape=(new_shape), dtype="float32")
new_func = relay.Function([x], new_body)
mod["my_func"] = new_func
```

### 패턴 3: 임시 함수 정리

```python
# Transform 중 임시 함수들 정리
temp_funcs = ["_temp1", "_temp2", "_helper"]

for name in temp_funcs:
    try:
        del mod[name]
    except ValueError:
        pass  # 없으면 무시
```

### 패턴 4: 안전한 삭제 헬퍼

```python
def safe_remove(mod, name):
    """
    안전하게 함수 삭제 (존재하지 않아도 에러 없음)
    """
    try:
        del mod[name]
        return True
    except ValueError:
        return False

# 사용
if safe_remove(mod, "my_func"):
    print("Deleted my_func")
else:
    print("my_func not found")
```

## 테스트 결과

### 삭제 전
```python
def @func1(%x: Tensor[(10), float32]) { ... }
def @func2(%x: Tensor[(20), float32]) { ... }
def @func3(%x: Tensor[(30), float32]) { ... }
```

### String으로 삭제: `del mod["func1"]`
```python
def @func2(%x: Tensor[(20), float32]) { ... }
def @func3(%x: Tensor[(30), float32]) { ... }
```

### GlobalVar로 삭제: `del mod[gv2]`
```python
def @func3(%x: Tensor[(30), float32]) { ... }
```

## 사용 케이스

### 1. Transform에서 불필요한 함수 제거
```python
class MyTransform:
    def transform(self, mod):
        # 처리 후 임시 함수들 제거
        for gv in mod.get_global_vars():
            if gv.name_hint.startswith("_tmp_"):
                del mod[gv]
        return mod
```

### 2. 모듈 병합 시 중복 제거
```python
# mod1과 mod2를 병합
for gv in mod2.get_global_vars():
    name = gv.name_hint
    if name in [g.name_hint for g in mod1.get_global_vars()]:
        del mod1[name]  # 중복 제거
    mod1[name] = mod2[gv]  # 추가
```

### 3. 선택적 함수 export
```python
# 특정 함수들만 남기고 나머지 삭제
keep = ["main", "imcflow_conv", "imcflow_relu"]

to_remove = []
for gv in mod.get_global_vars():
    if gv.name_hint not in keep:
        to_remove.append(gv.name_hint)

for name in to_remove:
    del mod[name]
```

## 비교: 다른 방법들

### IRModule.update()로 새 모듈 만들기
```python
# __delitem__ 대신 새 모듈 생성
new_mod = tvm.IRModule()
for gv in old_mod.get_global_vars():
    if should_keep(gv):
        new_mod[gv] = old_mod[gv]
```

**장점**: 원본 보존
**단점**: 메모리 사용 증가

### __delitem__ 사용
```python
# In-place 삭제
for name in to_remove:
    del mod[name]
```

**장점**: In-place, 메모리 효율적
**단점**: 원본 수정됨

## 핵심 포인트

1. **두 가지 방법**: String 또는 GlobalVar로 삭제
2. **에러 처리**: 존재하지 않으면 ValueError
3. **안전한 패턴**: 먼저 수집 → 나중에 삭제
4. **용도**: 필터링, 정리, 함수 교체
5. **main도 삭제 가능**: 특별한 보호 없음

## 예제 코드

```python
# 완전한 예제
import tvm
from tvm import relay

# 모듈 생성
mod = tvm.IRModule()

# 함수들 추가
for i in range(5):
    x = relay.var("x", shape=(10,), dtype="float32")
    func = relay.Function([x], relay.nn.relu(x))
    mod[f"func{i}"] = func

print(f"Before: {len(mod.get_global_vars())} functions")

# 짝수 함수만 삭제
to_remove = [f"func{i}" for i in range(0, 5, 2)]

for name in to_remove:
    del mod[name]

print(f"After: {len(mod.get_global_vars())} functions")
# Output: Before: 5 functions
#         After: 2 functions
```

이제 `del mod["function_name"]` 또는 `del mod[global_var]`로 모듈에서 함수를 제거할 수 있습니다!
