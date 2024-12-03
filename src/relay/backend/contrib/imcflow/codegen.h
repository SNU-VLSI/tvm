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

#ifndef TVM_RELAY_CONTRIB_IMCFLOW_CODEGEN_H_
#define TVM_RELAY_CONTRIB_IMCFLOW_CODEGEN_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <vector>
#include <unordered_map>


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
static tvm::Array<Expr> BindToCallNodeArgs(const std::vector<Expr>& args, const CallNode* cn);

inline size_t GetShape1DSize(const Type& type);

inline std::string GetShapeString(std::vector<int> shape);

static std::vector<std::string> Conv2d(const CallNode* call);
static std::vector<std::string> Dense(const CallNode* call);
static std::vector<std::string> Relu(const CallNode* call);
static std::vector<std::string> BatchNorm(const CallNode* call);

// should comply with src/runtime/contrib/imcflow/imcflow.cc
static std::vector<std::string> Add(const CallNode* call);
static std::vector<std::string> Multiply(const CallNode* call);

// TODO(@zhiics, @comaniac): This is a basic implementation. We should implement
// all utilities and make a base class for users to implement.
class CodegenIMCFLOW : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
 public:
  explicit CodegenIMCFLOW(const std::string& id) { this->ext_func_id_ = id; }

  std::vector<Output> VisitExprDefault_(const Object* op) final;
  std::vector<Output> VisitExpr_(const VarNode* node) final;
  std::vector<Output> VisitExpr_(const TupleNode* node) final;
  std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final;
  std::vector<Output> VisitExpr_(const ConstantNode* cn) final;
  std::vector<Output> VisitExpr_(const CallNode* call) final;

  std::string JIT(const std::vector<Output>& out);

 private:
  std::vector<std::string> GetArgumentNames(const CallNode* call);

  GenerateBodyOutput GenerateOpCall(const CallNode* call);

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller);

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& attribute_args);

  GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const std::vector<std::string>& attribute_args);

  /*! \brief The id of the external imcflow ext_func. */
  std::string ext_func_id_{""};
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The index of global constants. */
  int const_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls IMCFLOW kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using IMCFLOW kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The array declared to store the constant values. */
  std::string const_array_name_;
  /*! \brief The declaration of intermeidate buffers. */
  std::vector<std::string> buf_decl_;
  /*! \brief The variable name to constant mapping. */
  Array<String> const_vars_;

  friend class IMCFLOWModuleCodegen;
};

/*!
 * \brief The IMCFLOW codegen helper to generate wrapepr function calls of IMCFLOW
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class IMCFLOWModuleCodegen : public CSourceModuleCodegenBase {
 public:
  // Create a corresponding IMCFLOW function for the given relay Function.
  std::pair<std::string, Array<String>> GenIMCFLOWFunc(const Function& func);

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and imcflow specific ones. To make
   * linking simpiler, the IMCFLOW kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/imcflow folder.
   *
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override;

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
};

struct BuildOutput {
  std::string graph_json;
  runtime::Module mod;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
};

class IMCFLOWVirtualModuleCodegen {
 public:
  runtime::Module CreateLLVMModule(const ObjectRef& ref);
};

/*!
 * \brief Constant Updater for IMCFLOW JSON runtime
 *
 * Not all originally existing ConstantNode should be passed to JSON runtime.
 * Some of them may be skipped or change ordering. So we have to apply the same traversing through
 * the graph as IMCFLOWJSONSerializer.
 */
struct IMCFLOWConstantUpdater : public ConstantUpdater {
 public:
  IMCFLOWConstantUpdater(const std::string& symbol,
                         std::unordered_map<std::string, runtime::NDArray>* params)
      : ConstantUpdater("imcflow_" + symbol, params) {}
  using ConstantUpdater::VisitExpr_;

  void VisitExpr_(const CallNode* cn) final;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * produce collection of required constant NDArrays.
 */
Map<String, runtime::NDArray> IMCFLOWConstantUpdaterFunc(Expr expr, std::string symbol);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif // TVM_RELAY_CONTRIB_IMCFLOW_CODEGEN_H_