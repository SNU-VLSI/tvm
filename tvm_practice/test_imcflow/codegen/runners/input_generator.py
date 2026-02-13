"""
Shared input data generator for IMCFLOW tests.

This module generates deterministic test inputs that can be saved to files
and loaded by both Python (CPU validation) and C (hardware execution).
"""

import numpy as np
import os
from typing import Dict, Tuple, Optional, List
import tvm
from tvm import relay


class InputGenerator:
    """Generate and manage test input data for models"""

    def __init__(self, mod: Optional[tvm.IRModule] = None, known_keys: List[str] = None, seed=42):
        """
        Args:
            mod: TVM IRModule to extract input information from. If provided,
                 the generator will automatically determine input names and shapes.
            known_keys: Optional list of string of known parameter names
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.mod = mod
        self.input_info = None

        if mod is not None:
            self.input_info = self._extract_input_info(mod, known_keys)

    def _extract_input_info(self, mod: tvm.IRModule, known_keys: List[str] = None) -> Dict[str, Dict]:
        """
        Extract input variable information from the IRModule.

        Args:
            mod: TVM IRModule
            known_keys: Optional list of string of known parameter names

        Returns:
            Dictionary mapping input name to {'shape': tuple, 'dtype': str}
        """
        # Get the main function
        if "main" in mod:
            func = mod["main"]
        else:
            # If no main, get the first function
            func = list(mod.functions.values())[0]

        # Extract input parameters (those that are not in known_keys)
        input_info = {}
        for param in func.params:
            name = param.name_hint
            ttype = param.type_annotation

            if name in (known_keys or []): # skip known params
                continue

            if isinstance(ttype, relay.ty.TensorType):
                # Convert TVM shape to Python ints
                shape = []
                for dim in ttype.shape:
                    try:
                        shape.append(int(dim))
                    except Exception:
                        # Fallback if dynamic: leave as-is
                        shape.append(dim)

                input_info[name] = {
                    "shape": tuple(shape),
                    "dtype": ttype.dtype
                }

        return input_info

    def _generate_random_tensor(self, dtype: str, shape: Tuple) -> np.ndarray:
        """
        Generate a random tensor with appropriate range for the dtype.

        Args:
            dtype: Data type string (e.g., 'float32', 'int8', 'uint8')
            shape: Shape of the tensor

        Returns:
            Random numpy array
        """
        if dtype in ("float32", "float16", "float64"):
            return np.random.uniform(-1, 1, shape).astype(dtype)
        if dtype.startswith("int"):
            try:
                bits = int(dtype.replace("int", ""))
            except Exception:
                bits = 32
            if bits == 4:
                return np.random.randint(-8, 8, size=shape, dtype=np.int8)
            if bits == 8:
                return np.random.randint(-128, 128, size=shape, dtype=np.int8)
            if bits == 16:
                return np.random.randint(-32768, 32768, size=shape, dtype=np.int16)
            if bits == 32:
                return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
            if bits == 64:
                return np.random.randint(-2**63, 2**63 - 1, size=shape, dtype=np.int64)
            return np.random.randint(-2**31, 2**31, size=shape, dtype=np.int32)
        if dtype.startswith("uint"):
            try:
                bits = int(dtype.replace("uint", ""))
            except Exception:
                bits = 32
            if bits == 4:
                return np.random.randint(0, 16, size=shape, dtype=np.uint8)
            if bits == 8:
                return np.random.randint(0, 256, size=shape, dtype=np.uint8)
            if bits == 16:
                return np.random.randint(0, 2**16, size=shape, dtype=np.uint16)
            if bits == 32:
                return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
            if bits == 64:
                return np.random.randint(0, np.iinfo(np.uint64).max, size=shape, dtype=np.uint64)
            return np.random.randint(0, 2**32, size=shape, dtype=np.uint32)
        # Default float32 if unrecognized
        return np.random.uniform(-1, 1, shape).astype("float32")

    def generate_input(self, pattern: str = "random") -> Dict[str, np.ndarray]:
        """
        Generate input data automatically based on the model's input info.

        Args:
            pattern: Type of pattern to generate:
                     - 'random': Random values (default)
                     - 'ones': All ones
                     - 'zeros': All zeros
                     - 'linear': Linear index pattern

        Returns:
            Dictionary mapping input name to numpy array

        Raises:
            ValueError: If no mod was provided in constructor
        """
        if self.input_info is None:
            raise ValueError(
                "Cannot generate input automatically: no model provided. "
                "Please provide a mod parameter in the constructor."
            )

        inputs = {}
        for name, info in self.input_info.items():
            shape = info["shape"]
            dtype = info["dtype"]

            if pattern == "random":
                inputs[name] = self._generate_random_tensor(dtype, shape)
            elif pattern == "ones":
                inputs[name] = np.ones(shape, dtype=dtype)
            elif pattern == "zeros":
                inputs[name] = np.zeros(shape, dtype=dtype)
            elif pattern == "linear":
                # Create a linear index pattern
                total_elements = np.prod(shape)
                if dtype.startswith("float"):
                    arr = np.arange(total_elements, dtype=dtype)
                    arr = arr / total_elements  # Normalize to [0, 1)
                else:
                    arr = np.arange(total_elements, dtype=dtype)
                inputs[name] = arr.reshape(shape)
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

        return inputs

    def generate_resnet8_input(self, small_debug=False) -> Dict[str, np.ndarray]:
        """
        Generate input for ResNet8 CIFAR-10 model.
        
        Args:
            small_debug: If True, use 8x8 spatial size; otherwise 32x32
        
        Returns:
            Dictionary with 'model_input' -> numpy array
        """
        if small_debug:
            N, C, H, W = 1, 3, 8, 8
        else:
            N, C, H, W = 1, 3, 32, 32
        
        # Generate deterministic pattern that's easy to verify
        # Using a simple linear index pattern similar to C code
        input_data = np.zeros((N, C, H, W), dtype='int8')
        
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    # Simple pattern: constant value (easier to debug)
                    input_data[0, c, h, w] = 1
                    
                    # Alternative: Linear index pattern (uncomment if needed)
                    # idx = c * H * W + h * W + w
                    # input_data[0, c, h, w] = (idx % 128) - 64  # Range: [-64, 63]
        
        return {
            'model_input': input_data
        }
    
    def generate_mobilenet_input(self, small_debug=False) -> Dict[str, np.ndarray]:
        """Generate input for MobileNet model"""
        if small_debug:
            N, C, H, W = 1, 3, 32, 32
        else:
            N, C, H, W = 1, 3, 224, 224
        
        # Uniform random in quantized range [0, 255]
        input_data = np.random.randint(0, 256, size=(N, C, H, W), dtype='uint8')
        return {'input': input_data}
    
    def generate_one_relu_input(self) -> Dict[str, np.ndarray]:
        """Generate input for single ReLU test"""
        # Match the test configuration from models_for_test.py
        N, C, H, W = 1, 28, 4, 4

        # Generate deterministic pattern with linear index
        input_data = np.zeros((N, C, H, W), dtype='int16')
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    idx = c * H * W + h * W + w
                    input_data[0, c, h, w] = idx

        return {'input': input_data}

    def generate_one_conv_input(self) -> Dict[str, np.ndarray]:
        """Generate input for single convolution test"""
        # Match the test configuration from models_for_test.py
        N, C, H, W = 1, 28, 4, 4

        # Generate simple constant pattern (value = 1)
        input_data = np.ones((N, C, H, W), dtype='uint8')
        return {'conv_input': input_data}
    
    def save_to_files(self, input_dict: Dict[str, np.ndarray], output_dir: str):
        """
        Save input data to both .npy (Python) and .bin (C) formats.
        
        Args:
            input_dict: Dictionary of input name -> numpy array
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in input_dict.items():
            # Save as NumPy format (for Python)
            npy_path = os.path.join(output_dir, f"{name}.npy")
            np.save(npy_path, data)
            print(f"Saved {name} to {npy_path} (shape={data.shape}, dtype={data.dtype})")
            
            # Save as binary format (for C)
            bin_path = os.path.join(output_dir, f"{name}.bin")
            data.tofile(bin_path)
            print(f"Saved {name} to {bin_path} ({data.nbytes} bytes)")
            
            # Save metadata (shape, dtype) for C code
            meta_path = os.path.join(output_dir, f"{name}.meta.txt")
            with open(meta_path, 'w') as f:
                f.write(f"shape: {','.join(map(str, data.shape))}\n")
                f.write(f"dtype: {data.dtype}\n")
                f.write(f"nbytes: {data.nbytes}\n")
            print(f"Saved metadata to {meta_path}")
    
    @staticmethod
    def load_from_files(input_dir: str, input_name: str) -> np.ndarray:
        """
        Load input data from .npy file.
        
        Args:
            input_dir: Directory containing input files
            input_name: Name of the input (without extension)
        
        Returns:
            numpy array with input data
        """
        npy_path = os.path.join(input_dir, f"{input_name}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Input file not found: {npy_path}")
        
        data = np.load(npy_path)
        print(f"Loaded {input_name} from {npy_path} (shape={data.shape}, dtype={data.dtype})")
        return data


# Convenience functions for common use cases
def generate_and_save_resnet8_input(output_dir: str, small_debug=False):
    """Generate and save ResNet8 input to files"""
    gen = InputGenerator()
    inputs = gen.generate_resnet8_input(small_debug=small_debug)
    gen.save_to_files(inputs, output_dir)
    return inputs


def load_resnet8_input(input_dir: str) -> Dict[str, np.ndarray]:
    """Load ResNet8 input from files"""
    gen = InputGenerator()
    data = gen.load_from_files(input_dir, 'model_input')
    return {'model_input': data}


def load_one_relu_input(input_dir: str) -> Dict[str, np.ndarray]:
    """Load one ReLU input from files"""
    gen = InputGenerator()
    data = gen.load_from_files(input_dir, 'input')
    return {'input': data}


def load_one_conv_input(input_dir: str) -> Dict[str, np.ndarray]:
    """Load one Conv input from files"""
    gen = InputGenerator()
    data = gen.load_from_files(input_dir, 'conv_input')
    return {'conv_input': data}


if __name__ == "__main__":
    # Example usage: Generate test inputs for all models
    print("="*60)
    print("Generating test inputs...")
    print("="*60)
    
    gen = InputGenerator(seed=42)
    
    # ResNet8 (small debug)
    print("\n--- ResNet8 (small_debug) ---")
    resnet8_inputs = gen.generate_resnet8_input(small_debug=True)
    gen.save_to_files(resnet8_inputs, "./test_inputs/resnet8_small")
    
    # ResNet8 (full size)
    print("\n--- ResNet8 (full) ---")
    resnet8_full_inputs = gen.generate_resnet8_input(small_debug=False)
    gen.save_to_files(resnet8_full_inputs, "./test_inputs/resnet8_full")
    
    # MobileNet
    print("\n--- MobileNet ---")
    mobilenet_inputs = gen.generate_mobilenet_input(small_debug=False)
    gen.save_to_files(mobilenet_inputs, "./test_inputs/mobilenet")

    # One ReLU test
    print("\n--- One ReLU ---")
    onerelu_inputs = gen.generate_one_relu_input()
    gen.save_to_files(onerelu_inputs, "./test_inputs/one_relu")

    # One Conv test
    print("\n--- One Conv ---")
    oneconv_inputs = gen.generate_one_conv_input()
    gen.save_to_files(oneconv_inputs, "./test_inputs/one_conv")

    print("\n" + "="*60)
    print("✅ All test inputs generated successfully!")
    print("="*60)
