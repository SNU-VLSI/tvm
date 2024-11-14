# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_IMCFLOW)
  include_directories(src/relay/backend/contrib/imcflow)
  tvm_file_glob(GLOB IMCFLOW_RELAY_CONTRIB_SRC src/relay/backend/contrib/imcflow/*.cc)
  list(APPEND COMPILER_SRCS ${IMCFLOW_RELAY_CONTRIB_SRC})

  include_directories(src/runtime/contrib/imcflow)
  tvm_file_glob(GLOB IMCFLOW_CONTRIB_SRC src/runtime/contrib/imcflow/*.cc)
  list(APPEND RUNTIME_SRCS ${IMCFLOW_CONTRIB_SRC})
  message(STATUS "Build with IMCLFOW C source module")
endif(USE_IMCFLOW)
