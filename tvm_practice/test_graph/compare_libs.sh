#!/bin/bash

echo "=== Comparing lib.so files: cpp_graph vs crt_graph ==="
echo

echo "1. File type and basic info:"
file cpp_graph/lib.so crt_graph/lib.so
echo

echo "2. File sizes:"
ls -la cpp_graph/lib.so crt_graph/lib.so
echo

echo "3. Symbol table comparison:"
echo "--- cpp_graph/lib.so symbols ---"
objdump -T cpp_graph/lib.so | head -20
echo
echo "--- crt_graph/lib.so symbols ---"
objdump -T crt_graph/lib.so | head -20
echo

echo "4. TVM-related symbols in cpp_graph:"
objdump -T cpp_graph/lib.so | grep -E "(TVM|tvm)" | head -10
echo

echo "5. TVM-related symbols in crt_graph:"
objdump -T crt_graph/lib.so | grep -E "(TVM|tvm)" | head -10
echo

echo "6. System/Registry/Entry symbols in cpp_graph:"
objdump -T cpp_graph/lib.so | grep -E "(System|Registry|Entry)" | head -10
echo

echo "7. System/Registry/Entry symbols in crt_graph:"
objdump -T crt_graph/lib.so | grep -E "(System|Registry|Entry)" | head -10
echo

echo "8. Symbol count comparison:"
echo "cpp_graph symbols: $(objdump -T cpp_graph/lib.so | wc -l)"
echo "crt_graph symbols: $(objdump -T crt_graph/lib.so | wc -l)"
echo

echo "9. Complete symbol table diff:"
objdump -T cpp_graph/lib.so > cpp_symbols.txt
objdump -T crt_graph/lib.so > crt_symbols.txt
diff cpp_symbols.txt crt_symbols.txt
echo

echo "10. Section headers comparison:"
echo "--- cpp_graph sections ---"
readelf -S cpp_graph/lib.so | grep -E "(\.text|\.data|\.rodata|\.bss)" | head -10
echo
echo "--- crt_graph sections ---"
readelf -S crt_graph/lib.so | grep -E "(\.text|\.data|\.rodata|\.bss)" | head -10
echo

echo "11. String content comparison:"
echo "--- cpp_graph TVM strings ---"
strings cpp_graph/lib.so | grep -E "(TVM|tvm)" | head -10
echo
echo "--- crt_graph TVM strings ---"
strings crt_graph/lib.so | grep -E "(TVM|tvm)" | head -10
echo

echo "12. System/Registry/Entry strings in cpp_graph:"
strings cpp_graph/lib.so | grep -E "(System|Registry|Entry)" | head -10
echo

echo "13. System/Registry/Entry strings in crt_graph:"
strings crt_graph/lib.so | grep -E "(System|Registry|Entry)" | head -10
echo

echo "14. Complete string diff:"
strings cpp_graph/lib.so > cpp_strings.txt
strings crt_graph/lib.so > crt_strings.txt
diff cpp_strings.txt crt_strings.txt | head -20
echo

echo "15. TVM-related string differences:"
diff cpp_strings.txt crt_strings.txt | grep -E "(TVM|tvm|System|Registry|Entry)" | head -10
echo

echo "16. Hex dump comparison (first 10 lines):"
echo "--- cpp_graph hex dump ---"
hexdump -C cpp_graph/lib.so | head -10
echo
echo "--- crt_graph hex dump ---"
hexdump -C crt_graph/lib.so | head -10
echo

echo "17. ELF header comparison:"
echo "--- cpp_graph header ---"
readelf -h cpp_graph/lib.so | grep -E "(Size|Entry)"
echo
echo "--- crt_graph header ---"
readelf -h crt_graph/lib.so | grep -E "(Size|Entry)"
echo

echo "18. Section layout comparison:"
echo "--- cpp_graph sections ---"
readelf -S cpp_graph/lib.so | grep -E "\[[0-9]+\]" | tail -5
echo
echo "--- crt_graph sections ---"
readelf -S crt_graph/lib.so | grep -E "\[[0-9]+\]" | tail -5
echo

echo "=== Comparison complete ==="
echo "Summary:"
echo "- cpp_graph/lib.so: $(ls -lh cpp_graph/lib.so | awk '{print $5}')"
echo "- crt_graph/lib.so: $(ls -lh crt_graph/lib.so | awk '{print $5}')"
echo "- Symbol count: $(objdump -T cpp_graph/lib.so | wc -l) vs $(objdump -T crt_graph/lib.so | wc -l)"
echo "- String differences: $(diff cpp_strings.txt crt_strings.txt | wc -l) lines"

# Clean up temporary files
rm -f cpp_symbols.txt crt_symbols.txt cpp_strings.txt crt_strings.txt