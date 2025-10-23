"""
GlobalVar vs Direct Function Call 비교 테스트
"""
import tvm
from tvm import relay

print("=" * 80)
print("GlobalVar과 Direct Function Call의 차이")
print("=" * 80)

# 1. Direct Function Call (GlobalVar 없음)
print("\n[케이스 1] Direct Function Call - GlobalVar 없이 함수를 직접 호출")
print("-" * 80)

x = relay.var("x", shape=(10,), dtype="float32")
inner_func = relay.Function([x], relay.nn.relu(x))

# main 함수에서 inner_func를 직접 호출 (inline)
y = relay.var("y", shape=(10,), dtype="float32")
call_expr = relay.Call(inner_func, [y])  # inner_func를 직접 사용
main_func = relay.Function([y], call_expr)

print("Main function:")
print(main_func)
print("\n특징:")
print("  - inner_func가 main_func의 body 안에 inline으로 포함됨")
print("  - fn (%x: ...) { ... } 형태로 표시됨")
print("  - 모듈 없이 standalone으로 존재 가능")

# InferTypeLocal 시도
print("\nInferTypeLocal 시도:")
try:
    result = relay.transform.InferTypeLocal(main_func)
    print(f"✓ SUCCESS! Return type: {result.ret_type}")
except Exception as e:
    print(f"✗ FAILED: {e}")

# 2. GlobalVar를 사용한 Function Call
print("\n" + "=" * 80)
print("[케이스 2] GlobalVar Call - 모듈에 등록된 함수를 GlobalVar로 호출")
print("-" * 80)

# IRModule 생성 및 inner_func 등록
mod = tvm.IRModule()
inner_gv = relay.GlobalVar("inner_func")  # GlobalVar 생성
mod[inner_gv] = inner_func  # 모듈에 등록

# main 함수에서 GlobalVar를 통해 호출
y2 = relay.var("y", shape=(10,), dtype="float32")
call_expr2 = relay.Call(inner_gv, [y2])  # GlobalVar 사용!
main_func2 = relay.Function([y2], call_expr2)

print("Main function:")
print(main_func2)
print("\n특징:")
print("  - inner_func가 모듈에 별도로 등록됨")
print("  - @inner_func 형태로 GlobalVar 참조")
print("  - 모듈 컨텍스트가 필수!")
print("\nIRModule:")
print(mod)

# InferTypeLocal 시도 (실패 예상)
print("\nInferTypeLocal 시도 (standalone function):")
try:
    result = relay.transform.InferTypeLocal(main_func2)
    print(f"✓ SUCCESS! Return type: {result.ret_type}")
except Exception as e:
    print(f"✗ FAILED!")
    print(f"   이유: main_func2가 @inner_func를 참조하는데,")
    print(f"        InferTypeLocal은 모듈 컨텍스트가 없어서")
    print(f"        @inner_func의 타입을 알 수 없음!")

# IRModule 방식으로 해결
print("\nInferType with IRModule 시도:")
# 새로운 모듈 생성 (inner_func 포함)
mod_with_main = tvm.IRModule()
mod_with_main[inner_gv] = inner_func  # inner_func 먼저 등록
mod_with_main["main"] = main_func2    # main 등록
mod_with_main = relay.transform.InferType()(mod_with_main)
inferred_main = mod_with_main["main"]
print(f"✓ SUCCESS! Return type: {inferred_main.ret_type}")
print("   이유: 모듈 컨텍스트가 있어서 @inner_func를 찾을 수 있음")

# 3. 실제 imcflow 사용 사례
print("\n" + "=" * 80)
print("[케이스 3] 실제 IMCFlow 코드에서는?")
print("-" * 80)

print("""
IMCFlow transform에서 처리하는 코드:

def @main(%x: Tensor[(1, 64, 56, 56), float32]) {
  let %a = @imcflow_func1(%x);  <- GlobalVar!
  let %b = @imcflow_func2(%a);  <- GlobalVar!
  %b
}

def @imcflow_func1(%p: ...) { ... }
def @imcflow_func2(%q: ...) { ... }

이런 구조에서는:
1. @main이 @imcflow_func1, @imcflow_func2를 참조
2. GlobalVar들이 모듈에 등록되어 있음
3. InferTypeLocal을 main에만 적용하면?
   -> @imcflow_func1, @imcflow_func2를 찾을 수 없음!
   -> 에러 발생!

따라서:
- InferType(IRModule) 사용 필수!
- 모듈 전체를 넘겨서 모든 GlobalVar를 resolve해야 함
""")

print("\n" + "=" * 80)
print("결론:")
print("=" * 80)
print("""
1. Direct Function Call (inline):
   - InferTypeLocal ✓ 가능
   - 함수가 standalone으로 완전히 정의됨
   
2. GlobalVar Call (모듈 참조):
   - InferTypeLocal ✗ 불가능
   - 모듈 컨텍스트가 필요 (@inner_func를 찾아야 함)
   - InferType(IRModule) 사용 필수!

IMCFlow는 대부분 GlobalVar를 사용하므로 InferType(IRModule) 필수!
""")
