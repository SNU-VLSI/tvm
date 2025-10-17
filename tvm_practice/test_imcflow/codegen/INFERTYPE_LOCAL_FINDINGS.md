# InferTypeLocal 실험 결과 정리

## 핵심 발견사항

### 1. **InferTypeLocal은 FuncType을 반환한다!**
```python
func = relay.Function([x], y, ret_type=None)
result = relay.transform.InferTypeLocal(func)

# result의 타입: FuncType (Function이 아님!)
type(result)  # <class 'tvm.ir.type.FuncType'>

# 따라서 result.body는 존재하지 않음
# result.ret_type만 사용 가능
print(result.ret_type)  # Tensor[(1, 64, 56, 56), float32] ✓
```

### 2. **InferTypeLocal vs InferType 비교**

| 특성 | InferTypeLocal | InferType (IRModule) |
|------|---------------|---------------------|
| 반환 타입 | **FuncType** | **Function** |
| ret_type 접근 | ✓ 가능 | ✓ 가능 |
| body 접근 | ✗ 불가능 (FuncType) | ✓ 가능 |
| checked_type | ✗ body 없음 | ✓ 가능 |
| GlobalVar | ✗ 에러 발생 | ✓ 정상 동작 |
| 모듈 필요 | ✗ 불필요 | ✓ 필요 |
| 속도 | 빠름 | 약간 느림 |

### 3. **성공한 케이스**

모든 standalone 함수에서 정상 동작:
- ✓ 기본 함수 (relu, add 등)
- ✓ Parameter 업데이트 후 (shape 변경)
- ✓ Tuple 리턴
- ✓ TupleGetItem
- ✓ Let binding
- ✓ Nested function call (GlobalVar 아닌 경우)

### 4. **실패한 케이스**

- ✗ **GlobalVar 호출**: `InternalError: Check failed: (ret.defined()) is false`
  - GlobalVar는 모듈 컨텍스트가 필요하므로 InferTypeLocal 사용 불가

### 5. **실제 사용 제약사항**

```python
# 현재 _ImcflowFunctionParamUpdater 코드:
def visit_function(self, fn):
    # ... update params ...
    
    # InferTypeLocal은 사용할 수 없다!
    # 이유: ret_type만 반환하고, body에 접근 불가
    # 우리는 업데이트된 body도 필요함
    
    # 현재 IRModule 방식이 필수:
    mod = tvm.IRModule.from_expr(new_func)
    mod = relay.transform.InferType()(mod)
    inferred_func = mod["main"]
    
    # 이제 body와 ret_type 모두 사용 가능
    new_body = inferred_func.body  # ✓
    ret_type = inferred_func.ret_type  # ✓
```

## 결론

### ❌ InferTypeLocal을 사용할 수 없는 이유:
1. **FuncType 반환**: Function이 아니므로 body, attrs 등 접근 불가
2. **우리는 업데이트된 Function이 필요함**: body와 ret_type 모두 필요
3. **GlobalVar 지원 안 함**: 모듈 내 함수 호출 처리 불가

### ✓ 현재 IRModule 방식이 올바른 이유:
1. **완전한 Function 반환**: body, ret_type, attrs 모두 접근 가능
2. **GlobalVar 지원**: 모듈 컨텍스트 제공
3. **정확한 타입 추론**: 전체 모듈 고려

### 📝 사용 가이드라인:

**InferTypeLocal을 사용해야 하는 경우:**
- 단순히 return type만 알고 싶을 때
- Standalone 함수의 타입만 확인
- GlobalVar 의존성이 없을 때

**InferType (IRModule)을 사용해야 하는 경우:**
- 업데이트된 Function 전체가 필요할 때 (우리 케이스!)
- body나 다른 속성에 접근해야 할 때
- GlobalVar가 있을 때
- 모듈 전체 타입 추론이 필요할 때

## 코드 변경 불필요

**현재 구현이 정확합니다!**

```python
# python/tvm/relay/backend/contrib/imcflow/transform.py
def visit_function(self, fn):
    # ... parameter updates ...
    
    # 이 방식이 올바름! 변경하지 말 것
    mod = tvm.IRModule.from_expr(new_func)
    mod = relay.transform.InferType()(mod)
    inferred_func = mod["main"]
    
    # Function 전체를 얻을 수 있음
    return relay.Function(
        inferred_func.params,
        inferred_func.body,
        inferred_func.ret_type,  # InferType로 추론됨
        inferred_func.type_params,
        fn.attrs
    )
```

## 학습 내용

1. **InferTypeLocal ≠ InferType with local scope**
   - 이름이 혼란스럽지만, 완전히 다른 기능
   - InferTypeLocal: 타입만 추론 (FuncType 반환)
   - InferType: 전체 IR 타입 추론 (Function 반환)

2. **Type inference의 두 가지 목적:**
   - Type checking: InferTypeLocal으로 충분
   - IR transformation: InferType 필요 (우리 케이스)

3. **TVM IR의 계층 구조:**
   - FuncType: 타입 정보만 (signature)
   - Function: 완전한 IR 노드 (body + type + attrs)
   - IRModule: 여러 Function의 컨텍스트
