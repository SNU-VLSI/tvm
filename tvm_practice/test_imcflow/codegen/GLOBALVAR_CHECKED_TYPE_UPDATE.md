# GlobalVar checked_type 업데이트 문제 및 해결

## 문제 상황

```python
# IMCFlow transform 코드
for i in range(num_func):
    mod[function_names[i]] = self.update_imcflow_func_params(mod[function_names[i]])
    # 여기서 함수의 파라미터와 리턴 타입이 변경됨
    
    gv = mod.get_global_var(function_names[i])
    print(gv.checked_type)  # <- 아직 이전 타입!
```

### 왜 문제인가?

1. `mod[gv] = new_func`로 함수를 교체하면
   - 함수 자체는 업데이트됨 (파라미터, 리턴 타입 변경)
   - **하지만 GlobalVar의 cached checked_type은 업데이트 안 됨!**

2. GlobalVar는 함수를 "참조"하는 객체
   - 이전 InferType에서 계산된 타입 정보를 캐싱
   - 함수만 교체해도 GlobalVar는 캐시된 타입 유지

3. 다른 코드가 GlobalVar를 참조할 때
   - `@imcflow_func(%x)` 같은 호출
   - GlobalVar의 checked_type을 보고 타입 체크
   - **잘못된 타입 정보로 인해 에러 발생 가능**

## 테스트 결과

```python
# 함수 교체 전
gv.checked_type
# -> FuncType([], [TensorType([1, 64, 56, 56], float32)], 
#             TensorType([1, 64, 56, 56], float32))

# 함수 교체 (shape 변경)
new_func = Function([Var("x", shape=(1, 4, 56, 56, 16))], ...)
mod[gv] = new_func

gv.checked_type  # 아직 이전 타입!
# -> FuncType([], [TensorType([1, 64, 56, 56], float32)],  # <- 이전 shape!
#             TensorType([1, 64, 56, 56], float32))

# InferType 다시 실행
mod = relay.transform.InferType()(mod)

gv.checked_type  # 업데이트됨!
# -> FuncType([], [TensorType([1, 4, 56, 56, 16], float32)],  # <- 새 shape!
#             TensorType([1, 4, 56, 56, 16], float32))
```

## 해결 방법

### ✅ 정답: 함수 업데이트 후 InferType 다시 실행

```python
# IMCFlow transform 코드
for i in range(num_func):
    if function_names[i] == "main":
        continue
    elif ("Compiler" in mod[function_names[i]].attrs and 
          mod[function_names[i]].attrs["Compiler"] == "imcflow"):
        # 함수들 업데이트
        mod[function_names[i]] = self._mark_imcflow_function_boundaries(...)
        mod[function_names[i]] = self._mark_and_transform_imcflow_qconv(...)
        mod[function_names[i]] = self.update_imcflow_func_params(...)
        # 여기까지는 GlobalVar의 checked_type이 업데이트 안 됨!

# ✅ 핵심: 루프 후 InferType 다시 실행!
print("Running InferType to update GlobalVar checked_type...")
mod = relay.transform.InferType()(mod)

# 이제 모든 GlobalVar의 checked_type가 업데이트됨!
for i in range(num_func):
    if function_names[i] != "main":
        gv = mod.get_global_var(function_names[i])
        print(f"{function_names[i]} type: {gv.checked_type}")  # 새 타입!
```

### 왜 루프 후에 InferType을 실행하나?

1. **효율성**: 매번 실행하지 않고 한 번만 실행
   ```python
   # ✗ 비효율적
   for i in range(num_func):
       mod[gv] = new_func
       mod = relay.transform.InferType()(mod)  # 매번 실행 (느림!)
   
   # ✓ 효율적
   for i in range(num_func):
       mod[gv] = new_func
   mod = relay.transform.InferType()(mod)  # 한 번만 실행
   ```

2. **일관성**: 모든 함수를 업데이트한 후 전체 모듈의 타입을 한 번에 추론
   - 함수 간 의존성 고려
   - 전체 모듈 타입 일관성 보장

## 코드 변경 사항

### Before
```python
for i in range(num_func):
    # ... update functions ...
    mod[function_names[i]] = self.update_imcflow_func_params(...)
    print(mod.get_global_var(function_names[i]).checked_type)  # 잘못된 타입!

# Transform main
mod = self._insert_packing_unpacking(mod)
```

### After
```python
for i in range(num_func):
    # ... update functions ...
    mod[function_names[i]] = self.update_imcflow_func_params(...)
    # GlobalVar checked_type은 아직 업데이트 안 됨

# ✅ InferType 다시 실행하여 GlobalVar checked_type 업데이트
mod = relay.transform.InferType()(mod)

# Verify updated types
for i in range(num_func):
    if function_names[i] != "main":
        gv = mod.get_global_var(function_names[i])
        print(f"{function_names[i]} type: {gv.checked_type}")  # 올바른 타입!

# Transform main
mod = self._insert_packing_unpacking(mod)
```

## 핵심 포인트

1. **GlobalVar는 함수의 "참조"**
   - 함수를 직접 포함하지 않고 이름으로 참조
   - checked_type은 캐시된 값

2. **함수 교체만으로는 GlobalVar 타입이 안 바뀜**
   - `mod[gv] = new_func` ← 함수만 바뀜
   - `gv.checked_type` ← 여전히 이전 값

3. **InferType를 다시 실행해야 함**
   - `mod = relay.transform.InferType()(mod)`
   - 모든 GlobalVar의 checked_type 재계산
   - 함수 간 의존성 고려하여 타입 추론

4. **효율성을 위해 한 번만 실행**
   - 모든 함수 업데이트 후 마지막에 한 번
   - 매번 실행하면 성능 저하

## 참고: GlobalVar vs Function

```python
# IRModule 구조
mod = IRModule()
mod[gv1] = func1  # GlobalVar gv1이 func1을 참조
mod[gv2] = func2  # GlobalVar gv2가 func2를 참조

# GlobalVar는 "이름"
gv1.name_hint  # "imcflow_conv"

# Function은 "실제 코드"
func1.params   # [Var("x", ...)]
func1.body     # RelayExpr

# checked_type은 Function의 시그니처를 요약
gv1.checked_type  # FuncType([], [TensorType(...)], TensorType(...))
```

## 결론

**GlobalVar의 checked_type을 업데이트하려면:**
1. 함수를 교체: `mod[gv] = new_func`
2. **반드시 InferType 다시 실행**: `mod = relay.transform.InferType()(mod)`
3. 이제 `gv.checked_type`가 업데이트됨!

이것이 TVM Relay의 타입 시스템 동작 방식입니다!
