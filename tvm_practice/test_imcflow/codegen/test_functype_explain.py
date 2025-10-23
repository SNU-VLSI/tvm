"""
FuncType 구조 설명
"""
import tvm
from tvm import relay
from tvm.ir import TensorType, TupleType

print("=" * 80)
print("FuncType 구조 분석")
print("=" * 80)

# 예제 FuncType
func_type_str = """
I.FuncType(
    [],  # <- 1. Type parameters (제네릭 타입)
    [I.TensorType([1, 16, 28, 28], "int16")],  # <- 2. Input types (함수 파라미터)
    T.Tuple(  # <- 3. Return type (리턴 타입)
        I.TensorType([1, 16, 28, 28], "int16"),
        I.TensorType([16], "int16"),
        I.TensorType([16], "int16")
    )
)
"""

print("\nFuncType 구조:")
print(func_type_str)

print("\n" + "=" * 80)
print("각 부분 설명")
print("=" * 80)

print("\n1. 첫 번째 인자: [] (Type Parameters)")
print("-" * 80)
print("   의미: 제네릭 타입 파라미터 (템플릿 파라미터)")
print("   []  -> 제네릭이 없음 (일반 함수)")
print("   예시:")
print("     def func(%x: Tensor[...]) { ... }  <- 일반 함수")
print("     def func<T>(%x: T) { ... }         <- 제네릭 함수 (Relay에서 거의 안 씀)")

print("\n2. 두 번째 인자: [I.TensorType([1, 16, 28, 28], 'int16')] (Argument Types)")
print("-" * 80)
print("   의미: 함수 파라미터의 타입 리스트")
print("   [TensorType(...)] -> 파라미터가 1개")
print("   ")
print("   분해:")
print("     TensorType([1, 16, 28, 28], 'int16')")
print("     -> shape: (1, 16, 28, 28)")
print("     -> dtype: int16")
print("   ")
print("   대응하는 함수:")
print("     def func(%x: Tensor[(1, 16, 28, 28), int16]) {")
print("                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("                       이 부분의 타입!")
print("       ...")
print("     }")

print("\n3. 세 번째 인자: T.Tuple(...) (Return Type)")
print("-" * 80)
print("   의미: 함수의 리턴 타입")
print("   ")
print("   분해:")
print("     Tuple(")
print("       TensorType([1, 16, 28, 28], 'int16'),  <- Tuple의 첫 번째 원소")
print("       TensorType([16], 'int16'),             <- Tuple의 두 번째 원소")
print("       TensorType([16], 'int16')              <- Tuple의 세 번째 원소")
print("     )")
print("   ")
print("   대응하는 함수:")
print("     def func(%x: ...) -> (Tensor[(1,16,28,28), int16],")
print("                           Tensor[(16), int16],")
print("                           Tensor[(16), int16]) {")
print("                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("                        리턴 타입!")
print("       let %out = some_operation(%x)")
print("       let %stats1 = some_stats(%out)")
print("       let %stats2 = some_stats(%out)")
print("       (%out, %stats1, %stats2)  <- Tuple 리턴")
print("     }")

print("\n" + "=" * 80)
print("실제 예제로 확인")
print("=" * 80)

# 실제 함수 만들기
x = relay.var("x", shape=(1, 16, 28, 28), dtype="int16")

# Tuple을 리턴하는 함수
output = relay.add(x, relay.const(1, dtype="int16"))
stats1 = relay.mean(x, axis=[0, 2, 3])  # shape: (16,)
stats2 = relay.variance(x, axis=[0, 2, 3])  # shape: (16,)

# Cast stats to int16
stats1 = relay.cast(stats1, "int16")
stats2 = relay.cast(stats2, "int16")

tuple_output = relay.Tuple([output, stats1, stats2])
func = relay.Function([x], tuple_output)

print("\n생성된 함수:")
print(func)

# InferType로 타입 추론
from tvm import IRModule
mod = IRModule.from_expr(func)
mod = relay.transform.InferType()(mod)
inferred_func = mod["main"]

print("\n타입 추론 후:")
print(inferred_func)

# FuncType 확인
func_type = relay.transform.InferTypeLocal(func)
print("\nFuncType:")
print(func_type)
print(f"\nType parameters: {func_type.type_params}")
print(f"Argument types: {func_type.arg_types}")
print(f"Return type: {func_type.ret_type}")

if isinstance(func_type.ret_type, TupleType):
    print("\nReturn type is Tuple!")
    print(f"  Number of fields: {len(func_type.ret_type.fields)}")
    for i, field in enumerate(func_type.ret_type.fields):
        print(f"  Field {i}: {field}")

print("\n" + "=" * 80)
print("IMCFlow 컨텍스트에서")
print("=" * 80)
print("""
IMCFlow 함수가 여러 값을 리턴하는 경우:

def @imcflow_bn_relu(%input: Tensor[(1, 16, 28, 28), int16])
    -> (Tensor[(1, 16, 28, 28), int16],  # 출력 feature map
        Tensor[(16), int16],              # running mean
        Tensor[(16), int16]) {            # running variance
    
    let %out = imcflow.bn_relu(%input, ...)
    let %mean = %out.0   # TupleGetItem(0)
    let %var = %out.1    # TupleGetItem(1)
    (%out, %mean, %var)
}

이 함수의 FuncType:
  FuncType(
      [],  # 제네릭 없음
      [TensorType([1, 16, 28, 28], "int16")],  # %input 타입
      Tuple(  # 리턴 타입: 3개 원소의 Tuple
          TensorType([1, 16, 28, 28], "int16"),  # %out
          TensorType([16], "int16"),              # %mean
          TensorType([16], "int16")               # %var
      )
  )
""")

print("\n" + "=" * 80)
print("요약")
print("=" * 80)
print("""
FuncType(A, B, C):
  A: Type parameters (제네릭)      -> 거의 항상 []
  B: Argument types (파라미터)     -> [입력1_타입, 입력2_타입, ...]
  C: Return type (리턴 타입)       -> 출력_타입 (단일 or Tuple)

예시의 경우:
  파라미터: 1개 (Tensor[(1,16,28,28), int16])
  리턴:     3개 Tuple
            - Tensor[(1,16,28,28), int16]  # 메인 출력
            - Tensor[(16), int16]           # 통계1
            - Tensor[(16), int16]           # 통계2
""")
