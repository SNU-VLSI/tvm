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

#include "comp_op_matcher.h"
#include "device_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

/*!
 * \brief Replace var expr which bind with args of call node
 *
 * \param args vector of expression (contains vars or constant nodes)
 * \param cn call node which describe mapping of internal body vars with args
 * \return updated vector of expressions
 */
static tvm::Array<Expr> BindToCallNodeArgs(const std::vector<Expr>& args, const CallNode* cn) {
  tvm::Array<Expr> res;
  for (const auto& arg : args) {
    if (arg->IsInstance<ConstantNode>()) {
      res.push_back(arg);
    } else {
      auto body_params = cn->op.as<FunctionNode>()->params;
      auto found = std::find(body_params.begin(), body_params.end(), arg);
      ICHECK(found != body_params.end());
      auto idx = std::distance(body_params.begin(), found);
      res.push_back(cn->args[idx]);
    }
  }
  return res;
}

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

static std::vector<std::string> Conv2d(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  ICHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: O, G, Ph0, Pw0, Ph1, Pw1, Kh, Kw, Sh, Sw
  args.push_back(std::to_string(wshape[0]));
  args.push_back(std::to_string(conv2d_attr->groups));
  args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[2].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[3].as<IntImmNode>()->value));
  args.push_back(std::to_string(wshape[2]));
  args.push_back(std::to_string(wshape[3]));
  args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

  return args;
}

static std::vector<std::string> Dense(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, O
  args.push_back(std::to_string(ishape[0]));
  args.push_back(std::to_string(ishape[1]));
  args.push_back(std::to_string(wshape[0]));

  return args;
}

static std::vector<std::string> Relu(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  // Args: N, C, H, W
  args.push_back(GetShapeString(ishape));
  return args;
}

static std::vector<std::string> BatchNorm(const CallNode* call) {
  std::vector<std::string> args;
  const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: epsilon
  args.push_back(std::to_string(bn_attr->epsilon));

  return args;
}

// should comply with src/runtime/contrib/imcflow/imcflow.cc
#define IMCFLOW_BINARY_ADD 0
#define IMCFLOW_BINARY_MUL 1

static std::vector<std::string> Add(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  args.push_back(std::to_string(IMCFLOW_BINARY_ADD));
  // Args: H, W
  args.push_back(GetShapeString(ishape));
  return args;
}

static std::vector<std::string> Multiply(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  args.push_back(std::to_string(IMCFLOW_BINARY_MUL));
  // Args: H, W
  args.push_back(GetShapeString(ishape));
  return args;
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
  DLOG(INFO) << "IMCFLOW VarNode Visited";
  // ext_func_args_.push_back(GetRef<Var>(node));
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
  std::string const_symbol = "imcflow_" + ext_func_id_;
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
  DLOG(INFO) << "DEBUG_IMCFLOW (JIT)" << ret << std::endl;
  return ret;
}

// std::vector<std::string> CodegenIMCFLOW::GetArgumentNames(const CallNode* call) {
//   DLOG(INFO) << "IMCFLOW GetArgumentNames";
//   std::vector<std::string> arg_names;
//   for (size_t i = 0; i < call->args.size(); ++i) {
//     auto res = VisitExpr(call->args[i]);
//     for (const auto& out : res) {
//       arg_names.push_back(out.name);
//       DLOG(INFO) << out.name;
//     }
//   }
//   return arg_names;
// }

void CodegenIMCFLOW::GenerateOpCall(const CallNode* call) {
  DLOG(INFO) << "IMCFLOW GenerateOpCall Visited " << AsText(call->op, false);
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

  using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
  static const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
      {"nn.conv2d", {"imcflow_conv2d", Conv2d}},     {"nn.dense", {"imcflow_dense", Dense}},
      {"nn.relu", {"imcflow_relu", Relu}},           {"nn.batch_norm", {"imcflow_bn", BatchNorm}},
      {"nn.bias_add", {"imcflow_bias_add", Add}},    {"add", {"imcflow_binary_op", Add}},
      {"multiply", {"imcflow_binary_op", Multiply}},
  };

  const auto op_name = GetRef<Op>(op_node)->name;
  const auto iter = op_map.find(op_name);

  if (iter != op_map.end()) {
    // Visit operator and arguments
    std::vector<Output> res;
    for (const auto& arg : call->args) {
      VisitExpr(arg);
    }
    return;
  }
  LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
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

  std::ostringstream decl_stream;
  decl_stream << "(";
  // Make function call with input buffers when visiting arguments
  // ICHECK_GT(func_args.size(), 0);
  // std::ostringstream decl_stream;
  // decl_stream << "(" << func_args[0];
  // for (size_t i = 1; i < func_args.size(); ++i) {
  //   decl_stream << ", " << func_args[i];
  // }

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
  ICHECK(func.defined()) << "Input error: expect a Relay function.";

  // Record the external symbol for runtime lookup.
  auto sid = GetExtSymbol(func);

  CodegenIMCFLOW builder(sid);
  builder.VisitExpr(func->body); // visit the body and update the constant related variables

  auto out = builder.GenerateBody(func->body.as<CallNode>(), sid, {}, {});
  // std::vector<Output> out = {};

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

void IMCFLOWConstantUpdater::VisitExpr_(const CallNode* cn) {
  DLOG(INFO) << "IMCFLOW Constant Updater CallNode Visited";
  this->VisitSpan(cn->span);

  if (const auto* fn = cn->op.as<FunctionNode>()) {
    std::vector<Expr> args_loc;
    std::unordered_map<std::string, dmlc::any> attrs;
    auto root_cn = ParseComposite(*fn, &attrs, &args_loc);

    auto args = root_cn ? BindToCallNodeArgs(args_loc, cn) : cn->args;

    // Customized visit order of args
    for (const auto& arg : args) {
      this->VisitExpr(arg);
    }
  } else {
    // Original visit order of args
    for (auto arg : cn->args) {
      this->VisitExpr(arg);
    }
  }
}

// The external compiler/codegen tool
Map<String, runtime::NDArray> IMCFLOWConstantUpdaterFunc(Expr expr, std::string symbol) {
  // Visit all suitable constant nodes
  std::unordered_map<std::string, runtime::NDArray> res;
  IMCFLOWConstantUpdater const_updater(symbol, &res);
  const_updater(expr);

  // Convert to tvm::Map
  Map<String, runtime::NDArray> ret;
  for (const auto& kvp : res) ret.Set(kvp.first, kvp.second);
  return ret;
}

TVM_REGISTER_GLOBAL("relay.ext.imcflow.constant_updater").set_body_typed(IMCFLOWConstantUpdaterFunc);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
