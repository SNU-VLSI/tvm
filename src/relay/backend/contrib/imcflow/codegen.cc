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

#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"
#include "comp_op_matcher.h"

#include "../codegen_c/codegen_c.h"
#include "device_codegen.h"

#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>
#include "codegen.h"

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

// TODO(@zhiics, @comaniac): This is a basic implementation. We should implement
// all utilities and make a base class for users to implement.
std::vector<Output> CodegenIMCFLOW::VisitExprDefault_(const Object* op) {
  LOG(FATAL) << "IMCFLOW codegen doesn't support: " << op->GetTypeKey();
}

std::vector<Output> CodegenIMCFLOW::VisitExpr_(const VarNode* node) {
  LOG(INFO) << "IMCFLOW VarNode Visited";
  ext_func_args_.push_back(GetRef<Var>(node));
  Output output;
  output.name = node->name_hint();
  return {output};
}

std::vector<Output> CodegenIMCFLOW::VisitExpr_(const TupleNode* node) {
  LOG(INFO) << "IMCFLOW TupleNode Visited";
  std::vector<Output> outs;
  for (auto field : node->fields) {
    auto res = VisitExpr(field);
    ICHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
    outs.push_back(res[0]);
  }
  return outs;
}

std::vector<Output> CodegenIMCFLOW::VisitExpr_(const TupleGetItemNode* op) {
  LOG(INFO) << "IMCFLOW TupleGetItemNode Visited";
  auto res = VisitExpr(op->tuple);
  ICHECK_GT(res.size(), static_cast<size_t>(op->index));

  // Only keep the item we want for the child node.
  // FIXME(@comaniac): The other items should still be requried for the primary outputs.
  return {res[op->index]};
}

std::vector<Output> CodegenIMCFLOW::VisitExpr_(const ConstantNode* cn) {
  LOG(INFO) << "IMCFLOW ConstantNode Visited";
  Output output;
  // Get const: static_cast<float*>(imcflow_0_consts[0]->data)
  output.name = CreateDataReference(ext_func_id_, const_idx_);
  output.dtype = "float";

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

  return {output};
}

std::vector<Output> CodegenIMCFLOW::VisitExpr_(const CallNode* call) {
  LOG(INFO) << "IMCFLOW CallNode Visited";
  GenerateBodyOutput ret;
  if (const auto* func = call->op.as<FunctionNode>()) {
    ret = GenerateCompositeFunctionCall(func, call);
  } else {
    ret = GenerateOpCall(call);
  }

  buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
  ext_func_body_.push_back(ret.decl);
  return ret.outputs;
}

std::string CodegenIMCFLOW::JIT(const std::vector<Output>& out) {
  return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
}

std::vector<std::string> CodegenIMCFLOW::GetArgumentNames(const CallNode* call) {
  LOG(INFO) << "IMCFLOW GetArgumentNames";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < call->args.size(); ++i) {
    auto res = VisitExpr(call->args[i]);
    for (const auto& out : res) {
      arg_names.push_back(out.name);
      LOG(INFO) << out.name;
    }
  }
  return arg_names;
}

GenerateBodyOutput CodegenIMCFLOW::GenerateOpCall(const CallNode* call) {
  const auto* op_node = call->op.as<OpNode>();
  ICHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();

  using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
  static const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
      {"nn.conv2d", {"imcflow_conv2d", Conv2d}}, {"nn.dense", {"imcflow_dense", Dense}},
      {"nn.relu", {"imcflow_relu", Relu}},       {"nn.batch_norm", {"imcflow_bn", BatchNorm}},
      {"nn.bias_add", {"imcflow_bias_add", Add}},
      {"add", {"imcflow_binary_op", Add}},       {"multiply", {"imcflow_binary_op", Multiply}},
  };

  const auto op_name = GetRef<Op>(op_node)->name;
  const auto iter = op_map.find(op_name);
  if (iter != op_map.end()) {
    return GenerateBody(call, iter->second.first, iter->second.second(call));
  }

  LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
}

GenerateBodyOutput CodegenIMCFLOW::GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                  const CallNode* caller) {
  const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
  ICHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

  // using ExpectedOpType = std::vector<std::string>;
  // using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
  // static const std::map<std::string, std::tuple<int, ExpectedOpType, std::string, ArgFunType>> pat_map = {
  //     {"imcflow.conv2d_bias_relu", {2, {"nn.conv2d", "add", "nn.relu"}, "imcflow_fused_conv2d_bias_relu", Conv2d}},
  //     {"imcflow.conv2d_relu", {1, {"nn.conv2d", "nn.relu"}, "imcflow_fused_conv2d_relu", Conv2d}},
  //     // {"imcflow.conv2d_bias_add_bn_relu", {3, {"nn.conv2d", "nn.bias_add", "nn.batch_norm", "nn.relu"}, "imcflow_fused_conv2d_bias_add_bn_relu", Conv2d}},
  // };

  // auto info_ = pat_map.at(pattern_name.value());
  // const auto* call =
  //     GetRootCall(callee->body.as<CallNode>(), std::get<0>(info_), std::get<1>(info_));
  // return GenerateBody(call, std::get<2>(info_), GetArgumentNames(caller), std::get<3>(info_)(call));

  LOG(INFO) << "DEBUG_IMCFLOW Call body";
  auto callee_body = callee->body.as<CallNode>();

  LOG(INFO) << "DEBUG_IMCFLOW VisitExpr";
  auto out = this->VisitExpr(callee->body);

  const auto* call = GetRootCall(callee_body, "nn.conv2d");

  LOG(INFO) << "DEBUG_IMCFLOW GenerateBody";
  return GenerateBody(call, "imcflow_fused_func", GetArgumentNames(caller), Conv2d(call));

  LOG(FATAL) << "Unsupported composite function: " << AsText(call->op, false);
}

GenerateBodyOutput CodegenIMCFLOW::GenerateBody(const CallNode* root_call, const std::string& func_name,
                                const std::vector<std::string>& attribute_args) {
  return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args);
}

GenerateBodyOutput CodegenIMCFLOW::GenerateBody(const CallNode* root_call, const std::string& func_name,
                                const std::vector<std::string>& func_args,
                                const std::vector<std::string>& attribute_args) {
  LOG(INFO) << "IMCFLOW GenerateBody Enter";
  // Generate the Device Body
  std::string command = "mkdir -p ./imcflowlib";
  int err = system(command.c_str());
  ICHECK_EQ(err, 0) << "mkdir -p ./imcflowlib failed";

  DeviceCodegen device_codegen("./imcflowlib"); // TODO: change this directory
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

  GenerateBodyOutput ret;
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
    ret.buffers.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                          std::to_string(out_size) + ");");
    ret.outputs.push_back(output);
  }

  // Attach attribute arguments
  for (size_t i = 0; i < attribute_args.size(); ++i) {
    decl_stream << ", " << attribute_args[i];
  }
  decl_stream << ");";
  ret.decl = func_name + decl_stream.str();
  return ret;
}

// Create a corresponding IMCFLOW function for the given relay Function.
std::pair<std::string, Array<String>> IMCFLOWModuleCodegen::GenIMCFLOWFunc(const Function& func) {
  ICHECK(func.defined()) << "Input error: expect a Relay function.";

  // Record the external symbol for runtime lookup.
  auto sid = GetExtSymbol(func);

  LOG(INFO) << "DEBUG_IMCFLOW " << PrettyPrint(func) << std::endl;

  CodegenIMCFLOW builder(sid);
  auto out = builder.VisitExpr(func->body);
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
  IRModule func_module = WithAttrs(IRModule::FromExpr(func),
                                    {{tvm::attr::kExecutor, executor},
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
  PackedFunc f=RelayBuildModule.GetFunction("build_with_func_name");
  f(func_module, Array<tvm::Target>({target}), target, executor, runtime, wsmp, cmp, mod_name, func_name.value());

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
  LOG(INFO) << "IMCFLOW Constant Updater CallNode Visited";
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
