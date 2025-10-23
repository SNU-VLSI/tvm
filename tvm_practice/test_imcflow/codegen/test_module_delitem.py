"""
IRModule.__delitem__ 사용법 테스트
"""
import tvm
from tvm import relay

print("=" * 80)
print("IRModule.__delitem__ 사용법")
print("=" * 80)

# 1. IRModule 생성 및 함수 추가
print("\n1. IRModule에 함수들 추가:")
print("-" * 80)

mod = tvm.IRModule()

# 함수 1
x1 = relay.var("x", shape=(10,), dtype="float32")
func1 = relay.Function([x1], relay.nn.relu(x1))
gv1 = relay.GlobalVar("func1")
mod[gv1] = func1

# 함수 2
x2 = relay.var("x", shape=(20,), dtype="float32")
func2 = relay.Function([x2], relay.nn.relu(x2))
gv2 = relay.GlobalVar("func2")
mod[gv2] = func2

# 함수 3
x3 = relay.var("x", shape=(30,), dtype="float32")
func3 = relay.Function([x3], relay.nn.relu(x3))
gv3 = relay.GlobalVar("func3")
mod[gv3] = func3

print("Module with 3 functions:")
print(mod)

print("\nGlobalVars in module:")
for gv in mod.get_global_vars():
    print(f"  {gv.name_hint}")

# 2. String으로 삭제
print("\n2. String으로 함수 삭제 (del mod['func1']):")
print("-" * 80)

del mod["func1"]  # String 사용

print("After deleting 'func1':")
print(mod)

print("\nRemaining GlobalVars:")
for gv in mod.get_global_vars():
    print(f"  {gv.name_hint}")

# 3. GlobalVar로 삭제
print("\n3. GlobalVar로 함수 삭제 (del mod[gv2]):")
print("-" * 80)

del mod[gv2]  # GlobalVar 직접 사용

print("After deleting gv2:")
print(mod)

print("\nRemaining GlobalVars:")
for gv in mod.get_global_vars():
    print(f"  {gv.name_hint}")

# 4. 존재하지 않는 함수 삭제 시도
print("\n4. 존재하지 않는 함수 삭제 시도:")
print("-" * 80)

try:
    del mod["non_existent_func"]
    print("Success (no error)")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# 5. main 함수가 있는 경우
print("\n5. main 함수 삭제:")
print("-" * 80)

mod2 = tvm.IRModule()

# main 추가
x_main = relay.var("x", shape=(10,), dtype="float32")
main_func = relay.Function([x_main], relay.nn.relu(x_main))
mod2["main"] = main_func

# helper 추가
x_helper = relay.var("x", shape=(10,), dtype="float32")
helper_func = relay.Function([x_helper], relay.nn.relu(x_helper))
mod2["helper"] = helper_func

print("Module with main and helper:")
print(mod2)

# main 삭제
del mod2["main"]

print("\nAfter deleting main:")
print(mod2)

print("\nRemaining functions:")
for gv in mod2.get_global_vars():
    print(f"  {gv.name_hint}")

# 6. 실용적 사용 예: 필터링
print("\n6. 실용적 예제: 조건에 맞지 않는 함수 제거:")
print("-" * 80)

mod3 = tvm.IRModule()

# 여러 함수 추가 (일부는 imcflow attribute)
for i in range(5):
    x = relay.var("x", shape=(10,), dtype="float32")
    func = relay.Function([x], relay.nn.relu(x))
    
    if i % 2 == 0:
        func = func.with_attr("Compiler", "imcflow")
    
    gv = relay.GlobalVar(f"func{i}")
    mod3[gv] = func

print("Original module:")
for gv in mod3.get_global_vars():
    func = mod3[gv]
    compiler = func.attrs.get("Compiler", "none")
    print(f"  {gv.name_hint}: Compiler={compiler}")

# imcflow가 아닌 함수들 삭제
print("\nRemoving non-imcflow functions:")
to_remove = []
for gv in mod3.get_global_vars():
    func = mod3[gv]
    if "Compiler" not in func.attrs or func.attrs["Compiler"] != "imcflow":
        to_remove.append(gv.name_hint)

for name in to_remove:
    print(f"  Deleting {name}")
    del mod3[name]

print("\nFiltered module (only imcflow):")
for gv in mod3.get_global_vars():
    func = mod3[gv]
    compiler = func.attrs.get("Compiler", "none")
    print(f"  {gv.name_hint}: Compiler={compiler}")

# 7. 함수 삭제 후 재추가
print("\n7. 함수 삭제 후 같은 이름으로 재추가:")
print("-" * 80)

mod4 = tvm.IRModule()

# 초기 함수
x = relay.var("x", shape=(10,), dtype="float32")
old_func = relay.Function([x], relay.nn.relu(x))
mod4["my_func"] = old_func

print("Original function:")
print(mod4["my_func"])

# 삭제
del mod4["my_func"]

# 새로운 함수로 재추가
x_new = relay.var("x", shape=(20,), dtype="float32")
new_func = relay.Function([x_new], relay.add(x_new, x_new))
mod4["my_func"] = new_func

print("\nNew function with same name:")
print(mod4["my_func"])

# 8. 대량 삭제
print("\n8. 대량 함수 삭제:")
print("-" * 80)

mod5 = tvm.IRModule()

# 10개 함수 추가
for i in range(10):
    x = relay.var("x", shape=(10,), dtype="float32")
    func = relay.Function([x], relay.nn.relu(x))
    mod5[f"func{i}"] = func

print(f"Functions before deletion: {len(mod5.get_global_vars())}")

# 짝수 인덱스 함수들만 삭제
for i in range(0, 10, 2):
    del mod5[f"func{i}"]

print(f"Functions after deletion: {len(mod5.get_global_vars())}")
print("Remaining functions:")
for gv in mod5.get_global_vars():
    print(f"  {gv.name_hint}")

print("\n" + "=" * 80)
print("요약")
print("=" * 80)
print("""
IRModule.__delitem__ 사용법:

1. String으로 삭제:
   del mod["function_name"]

2. GlobalVar로 삭제:
   gv = relay.GlobalVar("function_name")
   del mod[gv]

3. 특징:
   - 존재하지 않는 함수 삭제 시 에러 없음 (조용히 무시)
   - main 함수도 삭제 가능
   - 삭제 후 같은 이름으로 재추가 가능

4. 실용적 패턴:
   # 조건부 필터링
   to_remove = []
   for gv in mod.get_global_vars():
       if some_condition(mod[gv]):
           to_remove.append(gv.name_hint)
   
   for name in to_remove:
       del mod[name]

5. 주의사항:
   - 삭제 중에는 get_global_vars() 결과가 변할 수 있음
   - 먼저 삭제할 이름들을 리스트로 수집 후 삭제
   - GlobalVar 객체는 삭제 후에도 유효 (참조만 제거됨)

6. 용도:
   - Transform에서 불필요한 함수 제거
   - 임시 함수 정리
   - 모듈 필터링
   - 함수 교체 (삭제 후 재추가)
""")
