"""
GlobalVar 개념 설명 - 가장 간단한 예제
"""
import tvm
from tvm import relay

print("=" * 80)
print("GlobalVar이란? - 모듈에 등록된 함수를 참조하는 이름")
print("=" * 80)

# 간단한 함수 하나 만들기
x = relay.var("x", shape=(10,), dtype="float32")
my_func = relay.Function([x], relay.nn.relu(x))

print("\n1. Relay Function 출력:")
print(my_func)

# IRModule에 함수 등록하기
mod = tvm.IRModule()
global_var = relay.GlobalVar("my_relu")  # 이게 GlobalVar!
mod[global_var] = my_func

print("\n2. IRModule에 등록 후:")
print(mod)
print("\n설명:")
print("  - @my_relu가 GlobalVar")
print("  - 모듈 안에서 함수를 가리키는 '이름'")

# 다른 함수에서 GlobalVar로 호출하기
y = relay.var("y", shape=(10,), dtype="float32")
call_with_globalvar = relay.Call(global_var, [y])  # GlobalVar 사용!
main_func = relay.Function([y], call_with_globalvar)

print("\n3. GlobalVar를 호출하는 main 함수:")
print(main_func)
print("\n설명:")
print("  - @my_relu(%y) 형태")
print("  - @가 붙으면 GlobalVar (모듈에서 찾아야 함)")

# 비교: GlobalVar 없이 직접 호출
call_direct = relay.Call(my_func, [y])  # 함수 직접 사용
main_func_direct = relay.Function([y], call_direct)

print("\n4. 비교: GlobalVar 없이 함수를 직접 inline으로 호출:")
print(main_func_direct)
print("\n설명:")
print("  - fn (%x: ...) { ... } 형태로 inline")
print("  - @가 없음 (모듈 필요 없음)")

print("\n" + "=" * 80)
print("핵심 차이점")
print("=" * 80)
print("""
[GlobalVar 사용]
fn (%y: ...) {
  @my_relu(%y)    <- @가 있음 = GlobalVar
}

특징:
- @my_relu를 resolve하려면 모듈 컨텍스트 필요
- InferTypeLocal로는 @my_relu가 뭔지 모름
- InferType(IRModule) 필수!

[Direct Call]
fn (%y: ...) {
  %0 = fn (%x: ...) { nn.relu(%x) };  <- inline으로 포함
  %0(%y)
}

특징:
- 모든 정보가 함수 안에 포함됨
- InferTypeLocal 가능
- 모듈 없이도 standalone으로 동작
""")

print("\n" + "=" * 80)
print("IMCFlow에서는?")
print("=" * 80)
print("""
IMCFlow transform은 모듈 전체를 변환:

Input IRModule:
  def @main(...) { ... @imcflow_conv(...) ... }  <- GlobalVar 사용!
  def @imcflow_conv(...) { ... }
  def @some_other_func(...) { ... }

모든 함수가 GlobalVar로 서로를 참조하므로:
-> InferType(IRModule) 사용 필수!
-> InferTypeLocal은 사용 불가!
""")
