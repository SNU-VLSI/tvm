/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/imcflow/codegen.cc
 * \brief Implementation of IMCFLOW codegen APIs.
 */
#include "codegen.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>

#include "device_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

inline std::string GetShapeString(std::vector<int> shape) {
  std::string v = "std::vector<long int>{";
  for (auto s : shape) {
    v += std::to_string(s) + ",";
  }
  v += "}";
  return v;
}

int CodegenIMCFLOW::VisitExprDefault_(const Object* op) {
  LOG(FATAL) << "IMCFLOW codegen doesn't support: " << op->GetTypeKey();
}

int CodegenIMCFLOW::VisitExpr_(const LetNode* op) {
  DLOG(INFO) << "IMCFLOW LetNode Visited";
  VisitExpr(op->var);
  VisitExpr(op->value);
  VisitExpr(op->body);
}

int CodegenIMCFLOW::VisitExpr_(const VarNode* node) {
  // ext_func_args_.push_back(GetRef<Var>(node));
  DLOG(INFO) << "IMCFLOW VarNode Visited";
}

int CodegenIMCFLOW::VisitExpr_(const TupleNode* node) {
  DLOG(INFO) << "IMCFLOW TupleNode Visited";
  for (auto field : node->fields) {
    VisitExpr(field);
  }
}

int CodegenIMCFLOW::VisitExpr_(const TupleGetItemNode* op) {
  DLOG(INFO) << "IMCFLOW TupleGetItemNode Visited";
  VisitExpr(op->tuple);
}

int CodegenIMCFLOW::VisitExpr_(const ConstantNode* cn) {
  DLOG(INFO) << "IMCFLOW ConstantNode Visited";

  // Generate the global variable for needed ndarrays
  if (const_array_name_.empty()) {
    const_array_name_ = CreateNDArrayPool(ext_func_id_);
    std::string checker = CreateInitChecker(ext_func_id_);
    ext_func_body_.insert(ext_func_body_.begin(), checker);
  }

  // Give the ndarray a unique name to ease the initialization of it at
  // runtime.
  std::string const_symbol = ext_func_id_;
  std::string const_var_name = CreateConstVar(const_symbol, const_idx_);
  const_vars_.push_back(const_var_name);
  const_idx_++;

  const auto* type_node = cn->checked_type().as<TensorTypeNode>();
  ICHECK(type_node);
  ICHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";
}

int CodegenIMCFLOW::VisitExpr_(const CallNode* call) {
  DLOG(INFO) << "IMCFLOW CallNode Visited";
  if (const auto* func = call->op.as<FunctionNode>()) {
    GenerateCompositeFunctionCall(func, call);
  } else {
    GenerateOpCall(call);
  }
}

std::string CodegenIMCFLOW::JIT(const std::vector<Output>& out) {
  std::string ret =
      JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  DLOG(INFO) << "DEBUG_IMCFLOW (JIT)\n" << ret << std::endl;
  return ret;
}

void CodegenIMCFLOW::GenerateOpCall(const CallNode* call) {
  DLOG(INFO) << "IMCFLOW GenerateOpCall Visited " << AsText(call->op, false);
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

  // Visit operator and arguments
  std::vector<Output> res;
  for (const auto& arg : call->args) {
    VisitExpr(arg);
  }
}

void CodegenIMCFLOW::GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                                  const CallNode* caller) {
  const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
  const auto hash_name = callee->GetAttr<runtime::String>(attr::kHash);
  DLOG(INFO) << "IMCFLOW GenerateCompositeFunctionCall Visited " << pattern_name.value()
            << " hash: " << hash_name.value();
  ICHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

  auto func = GetRef<Function>(callee);
  VisitExpr(func->body);
  for (const auto& arg : caller->args) {
    VisitExpr(arg);
  }
}

std::vector<Output> CodegenIMCFLOW::GenerateBody(const CallNode* root_call,
                                                const std::string& func_name,
                                                const std::vector<std::string>& func_args,
                                                const std::vector<std::string>& attribute_args) {
  DLOG(INFO) << "IMCFLOW GenerateBody Enter";
  // Generate the Device Body
  std::string command = "mkdir -p ./imcflowlib";
  int err = system(command.c_str());
  ICHECK_EQ(err, 0) << "mkdir -p ./imcflowlib failed";

  DeviceCodegen device_codegen("./imcflowlib");  // TODO: change this directory
  device_codegen.HandleDeviceCodeGeneration(func_name, func_args);

  // Generate Kernel Body

  // Make function call with input buffers when visiting arguments
  ICHECK_GT(func_args.size(), 0);
  std::ostringstream decl_stream;
  decl_stream << "(" << func_args[0];
  for (size_t i = 1; i < func_args.size(); ++i) {
    decl_stream << ", " << func_args[i];
  }

  // Analyze the output buffers
  std::vector<Type> out_types;
  if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
    auto type_node = root_call->checked_type().as<TupleTypeNode>();
    for (auto field : type_node->fields) {
      ICHECK(field->IsInstance<TensorTypeNode>());
      out_types.push_back(field);
    }
  } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
    ICHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
    out_types.push_back(root_call->checked_type());
  } else {
    LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
  }

  std::vector<Output> outputs;
  for (const auto& out_type : out_types) {
    this->PrintIndents();
    const std::string out = "buf_" + std::to_string(buf_idx_++);
    const auto out_size = GetShape1DSize(out_type);
    decl_stream << ", " << out;

    Output output;
    output.name = out;
    output.size = out_size;
    output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
    output.need_copy = true;
    buf_decl_.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                          std::to_string(out_size) + ");");
    outputs.push_back(output);
  }

  // Attach attribute arguments
  for (size_t i = 0; i < attribute_args.size(); ++i) {
    decl_stream << ", " << attribute_args[i];
  }
  decl_stream << ");";
  ext_func_body_.push_back(func_name + decl_stream.str());

  return outputs;
}

// Create a corresponding IMCFLOW function for the given relay Function.
std::pair<std::string, Array<String>> IMCFLOWModuleCodegen::GenIMCFLOWFunc(const Function& func) {
  auto func_node = func.as<FunctionNode>();
  ICHECK(func.defined()) << "Input error: expect a Relay function.";

  // Record the external symbol for runtime lookup.
  auto sid = GetExtSymbol(func);

  CodegenIMCFLOW builder(sid);
  builder.VisitExpr(func->body); // visit the body and update the constant related variables
  builder.ext_func_args_ = func_node->params;

  std::string func_name = "imcflow_fused_kernel";
  std::vector<std::string> func_args;
  for (const auto& arg : func_node->params) {
    func_args.push_back(arg->name_hint());
  }

  auto out = builder.GenerateBody(func->body.as<CallNode>(), func_name, func_args, {});

  code_stream_ << builder.JIT(out);

  return {sid, builder.const_vars_};
}

// Create a CSourceModule
runtime::Module IMCFLOWModuleCodegen::CreateCSourceModule(const ObjectRef& ref) {
  // Create headers
  code_stream_ << "#include <cstdint>\n";
  code_stream_ << "#include <cstdlib>\n";
  code_stream_ << "#include <cstring>\n";
  code_stream_ << "#include <vector>\n";
  code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
  code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
  code_stream_ << "#include <dlpack/dlpack.h>\n";
  // imcflow_kernel file is saved under src/runtime/contrib/imcflow so that we don't
  // expose it to ordinary users. To make export_library use it, users need to
  // pass -I${PATH_TO_TVM}/src/runtime/contrib
  code_stream_ << "#include <imcflow/imcflow_kernel.h>\n";
  code_stream_ << "using namespace tvm::runtime;\n";
  code_stream_ << "using namespace tvm::runtime::contrib;\n";
  code_stream_ << "\n";

  ICHECK(ref->IsInstance<FunctionNode>());
  auto res = GenIMCFLOWFunc(Downcast<Function>(ref));
  std::string code = code_stream_.str();
  DLOG(INFO) << "CreateCSourceModule - code:\n" << code << std::endl;
  String sym = std::get<0>(res);
  Array<String> variables = std::get<1>(res);

  // Create a CSource module
  const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
  ICHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
  // TODO(@manupa-arm): pass the function names to enable system-lib creation
  return (*pf)(code, "c", Array<String>{sym}, variables);
}

runtime::Module IMCFLOWVirtualModuleCodegen::CreateLLVMModule(const ObjectRef& ref) {
  // make module from ref
  relay::Executor executor = relay::Executor::Create("graph");
  relay::Runtime runtime = relay::Runtime::Create("cpp");
  WorkspaceMemoryPools wsmp = WorkspaceMemoryPools();
  ConstantMemoryPools cmp = ConstantMemoryPools();
  tvm::Target target("llvm");
  std::string mod_name("imcflow_cpu");
  // BuildOutput build_output;

  Function func = Downcast<Function>(ref);
  IRModule func_module =
      WithAttrs(IRModule::FromExpr(func), {{tvm::attr::kExecutor, executor},
                                           {tvm::attr::kRuntime, runtime},
                                           {tvm::attr::kWorkspaceMemoryPools, wsmp},
                                           {tvm::attr::kConstantMemoryPools, cmp}});

  tvm::runtime::Optional<String> func_name = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  // func_module->Update(tvm::GlobalVar("main"), func);
  std::cout << "DEBUG_IMCFLOW " << PrettyPrint(func_module) << std::endl;

  // executor_codegen_ = MakeExecutorCodegen("graph");
  // executor_codegen_->Init(nullptr, config_->primitive_targets);
  // executor_codegen_->Codegen(func_module, func, mod_name);
  // executor_codegen_->UpdateOutput(&ret_);
  // ret_.params = executor_codegen_->GetParams();

  // // use TVM built-in codegen to generate LLVM module
  std::string ext_name = "relay.build_module._BuildModule";
  auto pf = tvm::runtime::Registry::Get(ext_name);

  runtime::Module RelayBuildModule = (*pf)();
  PackedFunc f = RelayBuildModule.GetFunction("build_with_func_name");
  f(func_module, Array<tvm::Target>({target}), target, executor, runtime, wsmp, cmp, mod_name,
    func_name.value());

  return runtime::Module();
}

#ifdef USE_IMCFLOW_CSRC
/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module IMCFLOWCompiler(const ObjectRef& ref) {
  IMCFLOWModuleCodegen imcflow;
  return imcflow.CreateCSourceModule(ref);
}
#elif USE_IMCFLOW_VIRTUAL
/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module IMCFLOWCompiler(const ObjectRef& ref) {
  IMCFLOWVirtualModuleCodegen imcflow;
  return imcflow.CreateLLVMModule(ref);
}
#else
static_assert(false, "Either USE_IMCFLOW_CSRC or USE_IMCFLOW_VIRTUAL should be defined");
#endif

TVM_REGISTER_GLOBAL("relay.ext.imcflow").set_body_typed(IMCFLOWCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
