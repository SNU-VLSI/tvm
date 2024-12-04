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


#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

// Get the size of a 1D shape.
inline size_t GetShape1DSize(const Type& type);

// Get the string representation of a shape.
inline std::string GetShapeString(std::vector<int> shape);

/*!
 * \brief The IMCFLOW codegen class that generates C code for IMCFLOW kernels.
 */
class CodegenIMCFLOW : public MemoizedExprTranslator<int>, public CodegenCBase {
 public:
  explicit CodegenIMCFLOW(const std::string& id) { this->ext_func_id_ = id; }

  int VisitExprDefault_(const Object* op) final;
  int VisitExpr_(const LetNode* op) final;
  int VisitExpr_(const VarNode* node) final;
  int VisitExpr_(const TupleNode* node) final;
  int VisitExpr_(const TupleGetItemNode* op) final;
  int VisitExpr_(const ConstantNode* cn) final;
  int VisitExpr_(const CallNode* call) final;

  std::string JIT(const std::vector<Output>& out);

 private:

  void GenerateOpCall(const CallNode* call);

  void GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                    const CallNode* caller);

  std::vector<Output> GenerateBody(const CallNode* root_call, const std::string& func_name,
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
  std::vector<std::string> buf_decl_ = {};
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

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_CONTRIB_IMCFLOW_CODEGEN_H_