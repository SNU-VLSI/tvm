# Local Function 파라미터 업데이트 지원

## 변경 사항

`update_imcflow_func_params()`의 `visit_function()` 메서드를 수정하여 **local function의 파라미터와 리턴 타입도 재귀적으로 업데이트**하도록 개선했습니다.

## Local Function이란?

```python
# Global function (IRModule에 등록)
def @global_func(%x: ...) { ... }

# Local function (다른 함수의 body 안에 inline으로 존재)
fn (%outer_param: ...) {
  %local_fn = fn (%inner_param: ...) {  # <- Local function!
    nn.relu(%inner_param)
  };
  %local_fn(%outer_param)
}
```

**특징:**
- GlobalVar가 아닌 일반 Function 객체로 존재
- Outer function의 body 안에 중첩됨
- Let binding으로 변수에 바인딩될 수 있음
- 여러 레벨로 중첩 가능 (nested local functions)

## 문제 상황

### Before (이전 코드)

```python
def visit_function(self, fn):
    # ... parameter update ...
    
    # ✗ 문제: body를 그냥 bind만 함
    new_body = relay.bind(fn.body, param_map)
    
    # Local function은 업데이트 안 됨!
    return relay.Function(new_params, new_body, ...)
```

**문제점:**
1. Outer function의 파라미터만 업데이트
2. Body 안의 local function은 무시됨
3. Local function의 params/ret_type이 이전 값 유지

**예시:**
```python
# IMCFlow function with local helper
fn (%input: Tensor[(1, 64, 56, 56), float32]) {
  %helper = fn (%x: Tensor[(1, 64, 56, 56), float32]) {  # <- 업데이트 안 됨!
    nn.relu(%x)
  };
  %helper(%input)
}

# update_imcflow_func_params 실행 후
# -> outer %input은 (1, 4, 56, 56, 16)로 변경
# -> local %helper의 %x는 여전히 (1, 64, 56, 56)!  <- 문제!
```

### After (새 코드)

```python
def visit_function(self, fn):
    # ... parameter update ...
    
    # ✓ 해결: body를 재귀적으로 visit
    if len(param_map) == 0:
        # 파라미터 업데이트 없어도 local function을 위해 visit
        new_body = self.visit(fn.body)
    else:
        # 파라미터 substitution 후 visit
        substituted_body = relay.bind(fn.body, param_map)
        new_body = self.visit(substituted_body)
    
    # Local function도 자동으로 업데이트됨!
    return relay.Function(new_params, new_body, ...)
```

**개선점:**
1. `self.visit(fn.body)`가 재귀적으로 처리
2. Body 안의 모든 local function 탐색
3. 각 local function에 대해 `visit_function()` 재귀 호출
4. Local function의 params/ret_type도 업데이트

**예시:**
```python
# IMCFlow function with local helper
fn (%input: Tensor[(1, 64, 56, 56), float32]) {
  %helper = fn (%x: Tensor[(1, 64, 56, 56), float32]) {
    nn.relu(%x)
  };
  %helper(%input)
}

# update_imcflow_func_params 실행 후
# -> outer %input: (1, 64, 56, 56) -> (1, 4, 56, 56, 16)  ✓
# -> local %x:     (1, 64, 56, 56) -> (1, 4, 56, 56, 16)  ✓
# -> helper ret:   (1, 64, 56, 56) -> (1, 4, 56, 56, 16)  ✓
```

## 코드 비교

### Before
```python
def visit_function(self, fn):
    # ... gather consumers and update params ...
    
    # If no parameters were updated, return original function
    if len(param_map) == 0:
        return fn  # <- Local function 무시!
    
    # Update the function body with variable substitution
    new_body = relay.bind(fn.body, param_map)  # <- 단순 bind만!
    
    # ... create temp_func, run InferType ...
    return new_func
```

### After
```python
def visit_function(self, fn):
    # ... gather consumers and update params ...
    
    # Recursively visit the function body to update local functions
    if len(param_map) == 0:
        # No parameter updates, but still visit body for local functions
        new_body = self.visit(fn.body)  # <- 재귀 visit!
    else:
        # Apply parameter substitution first, then visit for local functions
        substituted_body = relay.bind(fn.body, param_map)
        new_body = self.visit(substituted_body)  # <- substitution + 재귀 visit!
    
    # Check if anything changed (params or body)
    if len(param_map) == 0 and new_body == fn.body:
        return fn  # <- 아무것도 안 바뀌면 return
    
    # ... create temp_func, run InferType ...
    return new_func
```

## 동작 방식

### 1. ExprMutator의 재귀 방문

`ExprMutator.visit()`는 자동으로 모든 하위 표현식을 방문:

```python
# Body 안의 구조
fn.body = Let(
    var="helper",
    value=Function([...], ...),  # <- Local function
    body=Call(helper, [...])
)

# self.visit(fn.body) 실행 시:
# 1. Let 방문 -> visit_let() 호출
# 2. Let.value 방문 -> Function이므로 visit_function() 호출!  <-
# 3. Let.body 방문 -> Call 방문
```

### 2. 중첩 처리

```python
# 3-level nested functions
fn_outer (%x1) {
    %fn_middle = fn (%x2) {
        %fn_inner = fn (%x3) {  # <- 가장 안쪽
            relu(%x3)
        };
        %fn_inner(%x2)
    };
    %fn_middle(%x1)
}

# visit_function(fn_outer) 호출 시:
# 1. fn_outer 처리 시작
# 2. self.visit(fn_outer.body) 
#    -> fn_middle 발견
#    -> visit_function(fn_middle) 재귀 호출
# 3. fn_middle 처리 시작
# 4. self.visit(fn_middle.body)
#    -> fn_inner 발견
#    -> visit_function(fn_inner) 재귀 호출
# 5. fn_inner 처리 (가장 안쪽부터 처리)
# 6. fn_middle 완료
# 7. fn_outer 완료
```

### 3. 테스트 결과

```
Visited function at depth 0  <- fn_outer
  Parameters: ['x1']
Visited function at depth 1  <- fn_middle
  Parameters: ['x2']
Visited function at depth 2  <- fn_inner
  Parameters: ['x3']

Total functions visited: 3  ✓
```

## IMCFlow 실제 사용 케이스

### Case 1: Helper Function

```python
def @imcflow_complex(%input: Tensor[(1, 64, 56, 56)]) {
    # Local helper for repeated operations
    %process = fn (%x: Tensor[(1, 64, 56, 56)]) {
        %0 = nn.relu(%x);
        %1 = nn.bias_add(%0, %bias);
        %1
    };
    
    %out1 = %process(%input);
    %out2 = %process(%out1);
    %process(%out2)
}

# update_imcflow_func_params 실행:
# -> %input: (1, 64, 56, 56) -> (1, 4, 56, 56, 16)
# -> %process의 %x: (1, 64, 56, 56) -> (1, 4, 56, 56, 16)  ✓
# -> %process ret: (1, 64, 56, 56) -> (1, 4, 56, 56, 16)    ✓
```

### Case 2: Closure for Configuration

```python
def @imcflow_with_config(%input: Tensor[...]) {
    # Configuration captured in closure
    let %scale = 0.5f;
    
    %scaler = fn (%x: Tensor[...]) {
        multiply(%x, %scale)  # Captures %scale
    };
    
    %scaler(%input)
}

# Local function도 업데이트되므로:
# -> %x의 shape가 packed layout으로 변경됨  ✓
```

## 핵심 포인트

1. **ExprMutator는 재귀적으로 모든 표현식을 방문**
   - `self.visit(fn.body)` 호출 시
   - Body 안의 모든 Function도 자동으로 `visit_function()` 호출

2. **Local function도 Function 객체**
   - GlobalVar가 아닐 뿐, 구조는 동일
   - params, body, ret_type 모두 가짐
   - 동일한 로직으로 업데이트 가능

3. **Param substitution 순서 중요**
   - 먼저 `relay.bind()`로 파라미터 치환
   - 그 다음 `self.visit()`로 local function 처리
   - 이 순서를 지켜야 올바른 변수 참조 유지

4. **효율성 고려**
   - `param_map`이 비어있어도 local function을 위해 visit
   - 하지만 아무것도 안 바뀌면 원본 함수 반환 (early return)

## 결론

**이제 `update_imcflow_func_params()`는:**
- ✓ Global function의 params/ret_type 업데이트
- ✓ Local function의 params/ret_type 업데이트 (NEW!)
- ✓ 중첩된 local function도 재귀적으로 처리 (NEW!)
- ✓ Let-bound local function도 처리 (NEW!)

**IMCFlow transform이 더 robust해졌습니다!**
