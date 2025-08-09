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
 * \file tvm/runtime/crt_config.h
 * \brief Template for CRT configuration, to be modified on each target.
 */
#ifndef TVM_RUNTIME_CRT_CRT_CONFIG_H_
#define TVM_RUNTIME_CRT_CRT_CONFIG_H_

#define TVM_CRT_LOG_LEVEL TVM_CRT_LOG_LEVEL_DEBUG
#define TVM_CRT_DEBUG 1
#define TVM_CRT_MAX_NDIM 6
#define TVM_CRT_MAX_ARGS 10
#define TVM_CRT_GLOBAL_FUNC_REGISTRY_SIZE_BYTES 512
#define TVM_CRT_MAX_REGISTERED_MODULES 2
#define TVM_CRT_MAX_PACKET_SIZE_BYTES 2048
#define TVM_CRT_MAX_STRLEN_DLTYPE 10
#define TVM_CRT_MAX_STRLEN_FUNCTION_NAME 120
#define TVM_CRT_MAX_STRLEN_PARAM_NAME 80

// Workspace size for CRT page allocator (used by host runner platform)
#ifndef TVM_WORKSPACE_SIZE_BYTES
#define TVM_WORKSPACE_SIZE_BYTES (8 * 1024 * 1024)
#endif

#endif  // TVM_RUNTIME_CRT_CRT_CONFIG_H_