# IMCFlow QConv Input 개수 변경 가이드

TVM의 IMCFlow QConv2D 연산의 input 개수를 변경하려면 다음 파일들을 수정해야 합니다.

## 현재 구조

현재 `imcflow_qconv`는 **2개의 input**을 받습니다:
- `data`: 입력 텐서
- `weight`: 가중치 텐서

## 수정해야 할 파일들

### 1. C++ Operator 등록 (핵심)

#### 📁 `/root/project/tvm/src/relay/op/nn/convolution.cc`

##### a) `set_num_inputs()` 변경 (라인 2085)
```cpp
RELAY_REGISTER_OP("nn.imcflow_qconv")
    .describe(R"code()code" TVM_ADD_FILELINE)
    .set_attrs_type<ImcflowQConv2DAttrs>()
    .set_num_inputs(2)  // ← 여기를 변경! (예: 3, 4 등)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    // 새로운 input 추가:
    // .add_argument("bias", "Tensor", "The bias tensor.")
    .set_support_level(2)
    .add_type_rel("ImcflowQConv2D", ImcflowQConv2DRel);
```

##### b) `MakeImcflowQConv()` 함수 수정 (라인 2051-2071)
```cpp
// 기존 코드 (2개 input):
inline Expr MakeImcflowQConv(Expr data, Expr weight, ...) {
  ...
  return Call(op, {data, weight}, Attrs(attrs), {});
}

// 수정 예시 (3개 input - bias 추가):
inline Expr MakeImcflowQConv(Expr data, Expr weight, Expr bias, ...) {
  ...
  return Call(op, {data, weight, bias}, Attrs(attrs), {});
}
```

##### c) `TVM_REGISTER_GLOBAL` 수정 (라인 2072-2080)
```cpp
// 기존:
TVM_REGISTER_GLOBAL("relay.op.nn._make.imcflow_qconv")
    .set_body_typed([](Expr data, Expr weight, Array<IndexExpr> strides, ...) {
      return MakeImcflowQConv(data, weight, strides, ...);
    });

// 수정 예시:
TVM_REGISTER_GLOBAL("relay.op.nn._make.imcflow_qconv")
    .set_body_typed([](Expr data, Expr weight, Expr bias, Array<IndexExpr> strides, ...) {
      return MakeImcflowQConv(data, weight, bias, strides, ...);
    });
```

##### d) Type Relation 함수 수정 (라인 1943-2020)
```cpp
bool ImcflowQConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);  // ← 변경! (input 개수 + 1)
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();
  // 새로운 input 추가:
  // const auto* bias = types[2].as<TensorTypeNode>();
  
  if (data == nullptr) return false;
  if (weight == nullptr) return false;
  // if (bias == nullptr) return false;
  
  ...
  // output type은 types[마지막] (예: types[3])
  reporter->Assign(types[3], TensorType(oshape, out_dtype));
  return true;
}
```

### 2. Python API

#### 📁 `/root/project/tvm/python/tvm/relay/op/nn/nn.py` (라인 3967-4006)

```python
# 기존:
def imcflow_qconv2d(
    data,
    weight,
    channels,
    in_channels,
    strides=(1, 1),
    ...
):
    ...
    return _make.imcflow_qconv(
        data,
        weight,
        strides,
        ...
    )

# 수정 예시:
def imcflow_qconv2d(
    data,
    weight,
    bias,  # 새로운 파라미터 추가
    channels,
    in_channels,
    strides=(1, 1),
    ...
):
    ...
    return _make.imcflow_qconv(
        data,
        weight,
        bias,  # 전달
        strides,
        ...
    )
```

### 3. Attributes 구조체 (필요 시)

#### 📁 `/root/project/tvm/include/tvm/relay/attrs/nn.h` (라인 1607-1700)

만약 새로운 input이 **Attrs**로 관리되어야 한다면 수정:
```cpp
struct ImcflowQConv2DAttrs : public tvm::AttrsNode<ImcflowQConv2DAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  ...
  bool const_packed_node;
  // 새로운 속성 추가 (필요 시):
  // bool use_bias;
  
  TVM_DECLARE_ATTRS(ImcflowQConv2DAttrs, "relay.attrs.ImcflowQConv2DAttrs") {
    ...
    TVM_ATTR_FIELD(const_packed_node)
        .set_default(false)
        .describe("Whether the weight is a constant packed node");
    // 새로운 속성 등록:
    // TVM_ATTR_FIELD(use_bias)
    //     .set_default(false)
    //     .describe("Whether to use bias");
  }
};
```

#### 📁 `/root/project/tvm/python/tvm/relay/op/op_attrs.py` (라인 690)
Python에서 접근하려면 여기도 업데이트:
```python
@tvm._ffi.register_object("relay.attrs.ImcflowQConv2DAttrs")
class ImcflowQConv2DAttrs(Attrs):
    """Attributes for nn.imcflow_qconv"""
    # 필요 시 Python용 헬퍼 메서드 추가
```

### 4. 관련 Transform 코드 (필요 시)

#### 📁 `/root/project/tvm/python/tvm/relay/backend/contrib/imcflow/transform.py`

imcflow_qconv2d를 호출하는 모든 곳을 찾아서 수정:
```python
# 예: 라인 248, 625, 2681 등
# 기존:
imcflow_qconv2d(input_new, weight_new, in_channels=..., channels=...)

# 수정:
imcflow_qconv2d(input_new, weight_new, bias_new, in_channels=..., channels=...)
```

#### 📁 `/root/project/tvm/python/tvm/relay/op/contrib/imcflow.py`

패턴 매칭에서 input 개수 확인:
```python
# 라인 1381 등에서 call.args 길이 체크
if hasattr(call.op, "name") and call.op.name == "nn.imcflow_qconv":
    # call.args[0] = data
    # call.args[1] = weight
    # call.args[2] = bias (새로 추가된 경우)
```

### 5. Strategy (필요 시)

#### 📁 `/root/project/tvm/python/tvm/relay/op/strategy/generic.py` (라인 2205)
```python
@override_native_generic_func("imcflow_qconv2d_strategy")
def imcflow_qconv2d_strategy(attrs, inputs, out_type, target):
    # inputs 개수 체크
    # inputs[0] = data
    # inputs[1] = weight
    # inputs[2] = bias (추가된 경우)
    ...
```

### 6. TOPI Implementation (필요 시)

#### 📁 `/root/project/tvm/python/tvm/topi/imcflow/qconv.py` (라인 5)
```python
# 기존:
def imcflow_qconv2d(
    Input, Weight, stride, padding, ...
):
    ...

# 수정:
def imcflow_qconv2d(
    Input, Weight, Bias, stride, padding, ...
):
    ...
```

## 수정 순서 (권장)

1. **C++ 코드 수정** (가장 중요)
   - `convolution.cc`: `set_num_inputs()`, `MakeImcflowQConv()`, `TVM_REGISTER_GLOBAL`, `ImcflowQConv2DRel`

2. **Python API 수정**
   - `nn.py`: `imcflow_qconv2d()` 함수 시그니처

3. **사용하는 곳 수정**
   - `transform.py`: imcflow_qconv2d 호출하는 모든 곳
   - 모델 코드: `resnet8_cifar.py` 등

4. **빌드 & 테스트**
   ```bash
   cd /root/project/tvm/build
   ../build.sh
   ```

5. **검증**
   ```python
   # 간단한 테스트
   import tvm
   from tvm import relay
   from tvm.relay.op.nn import imcflow_qconv2d
   
   data = relay.var("data", shape=(1, 3, 32, 32))
   weight = relay.var("weight", shape=(16, 3, 3, 3))
   # bias = relay.var("bias", shape=(16,))  # 새로 추가
   
   out = imcflow_qconv2d(data, weight, channels=16, in_channels=3)
   # out = imcflow_qconv2d(data, weight, bias, channels=16, in_channels=3)
   
   func = relay.Function([data, weight], out)
   # func = relay.Function([data, weight, bias], out)
   mod = tvm.IRModule.from_expr(func)
   mod = relay.transform.InferType()(mod)
   print(mod)
   ```

## 주의사항

### 1. Types 배열 크기
- `ImcflowQConv2DRel`에서 `ICHECK_EQ(types.size(), N)`
- N = input 개수 + 1 (output type 포함)
- 예: 2개 input → types.size() = 3
- 예: 3개 input → types.size() = 4

### 2. Call 생성 시 Args
```cpp
// {data, weight} → {data, weight, bias}
return Call(op, {data, weight, bias}, Attrs(attrs), {});
```

### 3. Pattern Matching
- `transform.py`, `imcflow.py`의 패턴 매칭에서 `call.args` 길이 변경
- 모든 visitor에서 args 인덱스 조정

### 4. Backward Compatibility
- 기존 코드 호환성 고려
- Optional parameter로 만들거나
- 새로운 op 이름 사용 (예: `nn.imcflow_qconv_v2`)

## 예시: Bias 추가하기

만약 bias를 3번째 input으로 추가한다면:

```diff
// convolution.cc
- .set_num_inputs(2)
+ .set_num_inputs(3)

- return Call(op, {data, weight}, Attrs(attrs), {});
+ return Call(op, {data, weight, bias}, Attrs(attrs), {});

- ICHECK_EQ(types.size(), 3);
+ ICHECK_EQ(types.size(), 4);

- const auto* weight = types[1].as<TensorTypeNode>();
+ const auto* weight = types[1].as<TensorTypeNode>();
+ const auto* bias = types[2].as<TensorTypeNode>();
```

```diff
# nn.py
def imcflow_qconv2d(
    data,
    weight,
+   bias,
    channels,
    ...
):
    return _make.imcflow_qconv(
        data,
        weight,
+       bias,
        strides,
        ...
    )
```

## 디버깅 팁

1. **빌드 에러**
   - C++ 컴파일 에러 → 함수 시그니처 불일치
   - `set_num_inputs()`와 실제 args 개수 확인

2. **런타임 에러**
   - `ICHECK_EQ` 실패 → types.size() 확인
   - "Argument count mismatch" → TVM_REGISTER_GLOBAL 파라미터 확인

3. **Type Inference 실패**
   - `ImcflowQConv2DRel` 내부 로직 확인
   - nullptr 체크 추가

4. **Transform 에러**
   - `call.args[i]` 인덱스 범위 확인
   - Pattern matching 업데이트

## 관련 파일 요약

| 파일 | 역할 | 수정 필요 |
|------|------|-----------|
| `src/relay/op/nn/convolution.cc` | Operator 등록 및 Type Relation | ✅ 필수 |
| `python/tvm/relay/op/nn/nn.py` | Python API | ✅ 필수 |
| `include/tvm/relay/attrs/nn.h` | Attributes 정의 | 선택적 |
| `python/tvm/relay/backend/contrib/imcflow/transform.py` | Transform 로직 | ✅ 필수 |
| `python/tvm/relay/op/contrib/imcflow.py` | Pattern matching | 선택적 |
| `python/tvm/relay/op/strategy/generic.py` | Strategy | 선택적 |
| `python/tvm/topi/imcflow/qconv.py` | TOPI 구현 | 선택적 |
| `tvm_practice/models/*.py` | 사용 예제 | ✅ 필수 |

## 참고

- TVM의 일반 conv2d는 3개 input 지원 (data, weight, bias)
- qnn.conv2d도 scale/zero_point 추가로 더 많은 input 지원
- 기존 패턴을 참고하여 확장 가능
