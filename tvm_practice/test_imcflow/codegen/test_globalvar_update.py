"""
GlobalVar의 checked_type 업데이트 테스트
"""
import tvm
from tvm import relay

print("=" * 80)
print("GlobalVar checked_type 업데이트 문제")
print("=" * 80)

# 1. 원본 함수 생성
x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
y = relay.nn.relu(x)
original_func = relay.Function([x], y)

# 2. IRModule에 등록
mod = tvm.IRModule()
gv = relay.GlobalVar("my_func")
mod[gv] = original_func

print("\n1. 원본 모듈:")
print(mod)
print(f"\nGlobalVar checked_type BEFORE InferType:")
try:
    print(f"  {gv.checked_type}")
except ValueError as e:
    print(f"  에러: {e}")
    print("  -> InferType 전에는 checked_type이 없음!")

# 3. InferType 실행
mod = relay.transform.InferType()(mod)

print(f"\nGlobalVar checked_type AFTER InferType:")
print(f"  {gv.checked_type}")
print(f"\n분석:")
print(f"  - 파라미터 타입: {gv.checked_type.arg_types}")
print(f"  - 리턴 타입: {gv.checked_type.ret_type}")

# 4. 함수를 업데이트 (파라미터 shape 변경)
print("\n" + "=" * 80)
print("함수 업데이트: shape (1,64,56,56) -> (1,4,56,56,16)")
print("=" * 80)

# 새로운 파라미터
new_x = relay.var("x", shape=(1, 4, 56, 56, 16), dtype="float32")
# body 업데이트
new_body = relay.bind(original_func.body, {x: new_x})
updated_func = relay.Function([new_x], new_body)

# 모듈 업데이트
print("\n방법 1: 직접 함수를 교체")
print("-" * 80)
mod[gv] = updated_func
print(mod)

print(f"\nGlobalVar checked_type (함수 교체 직후):")
print(f"  {gv.checked_type}")
print("  -> 여전히 이전 타입! InferType 안 됨!")

# 5. InferType 다시 실행
print("\n방법 2: InferType 다시 실행")
print("-" * 80)
mod = relay.transform.InferType()(mod)

print(f"\nGlobalVar checked_type (InferType 후):")
print(f"  {gv.checked_type}")
print(f"\n분석:")
print(f"  - 파라미터 타입: {gv.checked_type.arg_types}")
print(f"  - 리턴 타입: {gv.checked_type.ret_type}")
print("  -> 업데이트됨!")

# 6. 여러 함수가 있을 때
print("\n" + "=" * 80)
print("여러 함수가 있을 때")
print("=" * 80)

# 새로운 모듈
mod2 = tvm.IRModule()

# 함수1 (단순한 relu)
x1 = relay.var("x", shape=(10,), dtype="float32")
func1 = relay.Function([x1], relay.nn.relu(x1))
gv1 = relay.GlobalVar("func1")
mod2[gv1] = func1

print("\n원본 모듈:")
print(mod2)

# InferType
mod2 = relay.transform.InferType()(mod2)

print(f"\n타입 추론 후:")
print(f"  gv1.checked_type: {gv1.checked_type}")

# func1 업데이트
print("\n" + "-" * 80)
print("func1을 업데이트: shape (10,) -> (20,)")
print("-" * 80)

new_x1 = relay.var("x", shape=(20,), dtype="float32")
new_func1 = relay.Function([new_x1], relay.nn.relu(new_x1))
mod2[gv1] = new_func1

print(f"\ngv1.checked_type (교체 직후): {gv1.checked_type}")
print("  -> 여전히 (10,)!")

# InferType 다시 실행
mod2 = relay.transform.InferType()(mod2)

print(f"\ngv1.checked_type (InferType 후): {gv1.checked_type}")
print("  -> (20,)로 업데이트됨!")

# 7. 정답
print("\n" + "=" * 80)
print("결론: GlobalVar checked_type 업데이트 방법")
print("=" * 80)
print("""
문제:
  mod[gv] = updated_func  # 함수를 교체해도
  gv.checked_type         # GlobalVar의 타입은 자동으로 안 바뀜!

해결:
  mod[gv] = updated_func
  mod = relay.transform.InferType()(mod)  # 이걸 다시 실행!
  # 이제 gv.checked_type가 업데이트됨

왜 필요한가?
  - GlobalVar는 모듈에 등록된 함수를 "참조"하는 객체
  - 함수를 교체해도 GlobalVar 자체는 이전 타입 정보를 캐싱
  - InferType를 다시 실행해야 GlobalVar의 checked_type가 갱신됨

IMCFlow 코드에서:
  for i in range(num_func):
      mod[function_names[i]] = self.update_imcflow_func_params(mod[function_names[i]])
      # 여기서 GlobalVar의 checked_type는 아직 안 바뀜!
  
  # 해결: 루프 끝나고 InferType 다시 실행
  mod = relay.transform.InferType()(mod)
  # 이제 모든 GlobalVar의 checked_type가 업데이트됨!
""")

# 8. IMCFlow 시뮬레이션
print("\n" + "=" * 80)
print("IMCFlow 패턴 시뮬레이션")
print("=" * 80)

mod3 = tvm.IRModule()

# imcflow 함수들 생성
for i in range(3):
    x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
    func = relay.Function([x], relay.nn.relu(x))
    func = func.with_attr("Compiler", "imcflow")
    gv = relay.GlobalVar(f"imcflow_func{i}")
    mod3[gv] = func

mod3 = relay.transform.InferType()(mod3)

print("\n원본 모듈의 GlobalVar 타입:")
items = list(mod3.functions_items())
for gv, func in items:
    print(f"  {gv.name_hint}: {gv.checked_type}")

# 함수들 업데이트 (IMCFlow처럼)
print("\n함수들 업데이트 (shape 변경):")
items = list(mod3.functions_items())
for gv, func in items:
    if "Compiler" in func.attrs and func.attrs["Compiler"] == "imcflow":
        # 파라미터 업데이트
        old_param = func.params[0]
        new_param = relay.var("x", shape=(1, 4, 56, 56, 16), dtype="float32")
        new_body = relay.bind(func.body, {old_param: new_param})
        updated_func = relay.Function([new_param], new_body)
        updated_func = updated_func.with_attr("Compiler", "imcflow")
        
        mod3[gv] = updated_func
        print(f"  Updated {gv.name_hint}")
        print(f"    GlobalVar checked_type (직후): {gv.checked_type}")

# InferType 다시 실행 (핵심!)
print("\nInferType 다시 실행:")
mod3 = relay.transform.InferType()(mod3)

print("\n업데이트된 GlobalVar 타입:")
items = list(mod3.functions_items())
for gv, func in items:
    print(f"  {gv.name_hint}: {gv.checked_type}")
    print(f"    -> shape 변경 반영됨!")
