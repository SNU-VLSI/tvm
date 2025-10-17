"""
Local Function 파라미터 업데이트 테스트
"""
import tvm
from tvm import relay

print("=" * 80)
print("Local Function 파라미터 업데이트 테스트")
print("=" * 80)

# 1. Local function이 있는 함수 생성
print("\n1. Local function이 있는 imcflow 함수 구조:")
print("-" * 80)

# Outer function parameter
x_outer = relay.var("x_outer", shape=(1, 64, 56, 56), dtype="float32")

# Local function
x_inner = relay.var("x_inner", shape=(1, 64, 56, 56), dtype="float32")
inner_body = relay.nn.relu(x_inner)
local_func = relay.Function([x_inner], inner_body)

# Outer function body calls local function
call_local = relay.Call(local_func, [x_outer])
outer_body = relay.add(call_local, call_local)

# Create outer function
outer_func = relay.Function([x_outer], outer_body)

print("Original function with local function:")
print(outer_func)

# 2. IRModule에 등록하고 InferType
mod = tvm.IRModule()
gv = relay.GlobalVar("test_func")
mod[gv] = outer_func
mod = relay.transform.InferType()(mod)

print("\nAfter InferType:")
print(mod["test_func"])

# 3. 중첩된 local function 테스트
print("\n" + "=" * 80)
print("2. 중첩된 Local Function 테스트")
print("=" * 80)

# Level 3: Innermost function
x3 = relay.var("x3", shape=(1, 64, 56, 56), dtype="float32")
func3 = relay.Function([x3], relay.nn.relu(x3))

# Level 2: Middle function that calls func3
x2 = relay.var("x2", shape=(1, 64, 56, 56), dtype="float32")
call3 = relay.Call(func3, [x2])
func2 = relay.Function([x2], relay.add(call3, call3))

# Level 1: Outer function that calls func2
x1 = relay.var("x1", shape=(1, 64, 56, 56), dtype="float32")
call2 = relay.Call(func2, [x1])
func1 = relay.Function([x1], relay.multiply(call2, call2))

print("Nested local functions:")
print(func1)

# Register and infer type
mod2 = tvm.IRModule()
gv2 = relay.GlobalVar("nested_func")
mod2[gv2] = func1
mod2 = relay.transform.InferType()(mod2)

print("\nAfter InferType:")
print(mod2["nested_func"])

# 4. ExprMutator가 local function을 처리하는지 확인
print("\n" + "=" * 80)
print("3. ExprMutator로 Local Function 방문 확인")
print("=" * 80)

class LocalFunctionCounter(relay.ExprMutator):
    def __init__(self):
        super().__init__()
        self.function_count = 0
        self.function_depths = []
        self.current_depth = 0
    
    def visit_function(self, fn):
        self.function_count += 1
        self.function_depths.append(self.current_depth)
        print(f"  Visited function at depth {self.current_depth}")
        print(f"    Parameters: {[p.name_hint for p in fn.params]}")
        
        # Visit body with increased depth
        self.current_depth += 1
        new_body = self.visit(fn.body)
        self.current_depth -= 1
        
        # Return original function (no modification)
        return fn

counter = LocalFunctionCounter()
counter.visit(func1)

print(f"\nTotal functions visited: {counter.function_count}")
print(f"Function depths: {counter.function_depths}")
print(f"Expected: 3 functions (outer + 2 local)")

# 5. Let binding에 있는 local function
print("\n" + "=" * 80)
print("4. Let Binding의 Local Function 테스트")
print("=" * 80)

# Local function
x_local = relay.var("x_local", shape=(10,), dtype="float32")
local_fn = relay.Function([x_local], relay.nn.relu(x_local))

# Bind to a variable
local_fn_var = relay.var("my_func")

# Main function parameter
x_main = relay.var("x_main", shape=(10,), dtype="float32")

# Use the local function via Let binding
call_result = relay.Call(local_fn_var, [x_main])
let_body = relay.add(call_result, call_result)
let_expr = relay.Let(local_fn_var, local_fn, let_body)

main_func = relay.Function([x_main], let_expr)

print("Function with Let-bound local function:")
print(main_func)

# Count functions
counter2 = LocalFunctionCounter()
counter2.visit(main_func)

print(f"\nTotal functions visited: {counter2.function_count}")
print(f"Expected: 2 functions (main + local)")

# 6. 실제 패턴: IMCFlow에서 볼 수 있는 구조
print("\n" + "=" * 80)
print("5. IMCFlow 실제 패턴 시뮬레이션")
print("=" * 80)

# Helper function (local)
def create_helper_func():
    x = relay.var("x", shape=(1, 64, 56, 56), dtype="float32")
    y = relay.nn.relu(x)
    z = relay.add(y, relay.const(1.0, dtype="float32"))
    return relay.Function([x], z)

# Main imcflow function with local helper
x_main = relay.var("input", shape=(1, 64, 56, 56), dtype="float32")
helper = create_helper_func()

# Use helper multiple times
result1 = relay.Call(helper, [x_main])
result2 = relay.Call(helper, [result1])
result3 = relay.Call(helper, [result2])

imcflow_func = relay.Function([x_main], result3)
imcflow_func = imcflow_func.with_attr("Compiler", "imcflow")

print("IMCFlow function with local helper:")
print(imcflow_func)

# Register
mod3 = tvm.IRModule()
gv3 = relay.GlobalVar("imcflow_with_local")
mod3[gv3] = imcflow_func
mod3 = relay.transform.InferType()(mod3)

print("\nAfter InferType:")
print(mod3["imcflow_with_local"])

# Count functions
counter3 = LocalFunctionCounter()
counter3.visit(mod3["imcflow_with_local"])

print(f"\nTotal functions visited: {counter3.function_count}")
print(f"Expected: 2 (main + local helper)")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)
print("""
1. Local function은 outer function의 body 안에 inline으로 존재
   - Function 타입의 값으로 존재 (not GlobalVar)
   
2. ExprMutator.visit_function()은 local function도 자동으로 방문
   - fn.body를 visit할 때 재귀적으로 처리됨
   
3. Local function도 파라미터와 리턴 타입을 가짐
   - update_imcflow_func_params가 visit_function을 오버라이드하면
   - Local function도 자동으로 업데이트됨!
   
4. Let binding의 local function도 처리됨
   - Let의 value가 Function이면 visit_function 호출
   
5. IMCFlow에서:
   - Global imcflow function에 local helper가 있을 수 있음
   - 새로운 구현은 이런 local function의 params/ret_type도 업데이트
   - self.visit(fn.body)가 재귀적으로 처리
""")
