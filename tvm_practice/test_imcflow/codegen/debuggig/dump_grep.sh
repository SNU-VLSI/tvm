#!/bin/bash
grep -ir "imcflow.IMCE.1.1.send_fifo.0 is pushed" ./now.debug.log > mmquant.dump
grep -ir "line_buffer push called"                ./now.debug.log > linebuffer_in.dump
grep -ir "2.1.*OP_DWCONV: RESULT VALID!"               ./now.debug.log > dwconv_result_2.dump
grep -ir "3.1.*OP_DWCONV: RESULT VALID!"               ./now.debug.log > dwconv_result_3.dump
