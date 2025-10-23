# cmake -DCMAKE_INCLUDE_PATH=$CONDA_PREFIX/include ..
cmake ..
cmake --build . --parallel 64