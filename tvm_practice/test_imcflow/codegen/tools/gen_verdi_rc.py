#!/usr/bin/env python3
"""
Verdi Signal Preference (.rc) File Generator for IMCFlow RTL Simulation

Generates .rc files for Verdi waveform viewer with signal groups for
specified IMCE/inode grid coordinates. Supports both standalone tb_imcflow
and gem5 testbench hierarchies.

Usage:
    # Generate for specific IMCE nodes
    python gen_verdi_rc.py --nodes 0,1 1,2 3,3 -o signals.rc --fsdb path/to/file.fsdb

    # Generate for all nodes from a test directory (reads hw_node_map.txt)
    python gen_verdi_rc.py --test-dir ../ds_cnn_full_pretrained_evl -o signals.rc

    # Generate for an NxM grid (all IMCE nodes, col>=1)
    python gen_verdi_rc.py --grid 4x5 -o signals.rc --fsdb path/to/file.fsdb

    # Select specific signal groups
    python gen_verdi_rc.py --nodes 0,1 --groups ctrl router vpu -o signals.rc

    # Use gem5 testbench hierarchy
    python gen_verdi_rc.py --nodes 0,1 --tb gem5 --fsdb path/to/file.fsdb -o signals.rc

===============================================================================
Verdi .rc File Syntax Reference
===============================================================================

The .rc file is a plain-text Verdi nWave session file. Key directives:

  HEADER
  ------
  Magic 271485                          ; Required magic number (identifies Verdi .rc)
  Revision Verdi_R-2020.12-SP1         ; Verdi version string

  FSDB FILE
  ---------
  openDirFile -d / "" "path/to.fsdb"   ; Open waveform database
  activeDirFile "" "path/to.fsdb"      ; Set active file for subsequent addSignal

  WINDOW / VIEW
  -------------
  viewPort <x> <y> <w> <h> <sigW> <valW>  ; Window geometry
  signalSpacing <px>                       ; Vertical spacing between signals
  zoom <start> <end>                       ; Time range (in timeunit)
  cursor <time>                            ; Cursor position
  marker <time>                            ; Marker position
  top <row_index>                          ; First visible signal row
  curSTATUS ByChange|ByValue               ; Search mode

  SIGNAL GROUPS (collapsible sections in waveform)
  ------------------------------------------------
  addGroup "<name>"                     ; New group, expanded by default
  addGroup "<name>" -e FALSE            ; New group, collapsed by default
  addGroup "<name>" -c ID_YELLOW5 -e FALSE  ; With colored label
  ; (signals listed after addGroup belong to it until next addGroup)

  SUB-GROUPS (nested inside a group)
  ----------------------------------
  addSubGroup "<name>"                  ; Start a sub-group
  endSubGroup "<name>"                  ; End a sub-group

  ADDING SIGNALS
  --------------
  ; Full hierarchical path — also sets the "current scope":
  addSignal -h 15 /tb/top/mod/sub/signal_name[7:0]

  ; Short name reusing the current scope (set by previous full-path signal):
  addSignal -h 15 -holdScope another_signal
  addSignal -h 15 -holdScope bus[31:0]

  ; Display as unsigned:
  addSignal -h 15 -UNSIGNED -holdScope enum_signal[5:0]

  ; Expanded bus (shows individual bits):
  addSignal -expanded -h 15 -holdScope bus[3:0]

  Options:
    -h <height>       Signal row height in pixels (typically 15)
    -holdScope        Reuse scope from previous full-path addSignal
    -UNSIGNED         Display value as unsigned integer
    -expanded         Expand bus to show individual bits

  HIERARCHY PATH FORMAT
  ---------------------
  /testbench/module/genvar[index]/sub_module/interface/signal[MSB:LSB]

  For IMCFlow, there are two testbench hierarchies:
    standalone:  /tb_imcflow/u_imcflow/u_imcflow_impl/core_row[R]/core_col[C]/...
    gem5:        /testbench_imcflow_gem5/u_imcflow_with_axi/u_imcflow_impl/core_row[R]/core_col[C]/...

  Node types by column:
    col == 0:  inode   -> core_row[R]/core_col[0]/inode/u_intf_node/...
    col >= 1:  imce    -> core_row[R]/core_col[C]/imce_node/imce/...

  Router is a sibling of the node module:
    inode router:  core_row[R]/core_col[0]/inode/u_router/...
    imce router:   core_row[R]/core_col[C]/imce_node/u_router/...

  FOOTER SECTIONS (optional, for Verdi internal state)
  ----------------------------------------------------
  COMPLEX_EVENT_BEGIN / COMPLEX_EVENT_END
  GETSIGNALFORM_SCOPE_HIERARCHY_BEGIN / GETSIGNALFORM_SCOPE_HIERARCHY_END
  FILTER_SIGNAL_BEGIN / FILTER_SIGNAL_END

===============================================================================
How to Add New Signals
===============================================================================

Each signal group is defined as a function (e.g., signals_imce_ctrl) that
returns a list of signal entries. To add signals:

1. ADD A SIGNAL TO AN EXISTING GROUP
   Find the function for the group (e.g., signals_vpu for VPU signals) and
   append entries to its `signals` list using these helpers:

     _fsig(full_path)        Full hierarchical path. Sets the scope for
                              subsequent _sig() calls.
                              Example: _fsig(f"{base}/ctrl_ex/opcode[5:0]")

     _sig(name)              Signal name relative to the current scope
                              (set by the last _fsig). Uses -holdScope.
                              Example: _sig("ready")

     _sig(name, "UNSIGNED")  Same as above but displayed as unsigned.
                              Example: _sig("state[1:0]", "UNSIGNED")

     _sub(name)              Start a sub-group (collapsible section).
     _endsub(name)           End a sub-group.

   Example — add bshr_set_valid to the hazard_detector group:

     def signals_hazard_detector(tb, row, col):
         base = f"{imce_path(tb, row, col)}/u_imce_ctrl/u_hazard_detector"
         signals = [
             ...existing signals...
             _sig("bshr_set_valid"),          # <-- new signal (same scope)
         ]

   Example — add signals from a new sub-module with a sub-group:

     signals = [
         ...
         _sub("new_interface"),                             # collapsible section
         _fsig(f"{base}/new_interface/valid"),              # full path (sets scope)
         _sig("ready"),                                     # relative to new scope
         _sig("data[63:0]"),
         _endsub("new_interface"),
     ]

2. ADD A NEW SIGNAL GROUP
   a) Define a new function following the pattern:

        def signals_my_module(tb, row, col):
            base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_my_module"
            signals = [
                _fsig(f"{base}/clk_i"),
                _sig("rstn_i"),
                _sig("some_signal[7:0]"),
            ]
            return [("my_module", signals)]

   b) Register it in the SIGNAL_GROUPS dict:

        SIGNAL_GROUPS = {
            ...
            "my_module": signals_my_module,
        }

   c) Add to DEFAULT_IMCE_GROUPS (or DEFAULT_INODE_GROUPS) if it should be
      included by default:

        DEFAULT_IMCE_GROUPS = [..., "my_module"]

   Now usable via: --groups my_module
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict


# ============================================================================
# Testbench hierarchy prefixes
# ============================================================================
TB_PREFIXES = {
    "standalone": "/tb_imcflow/u_imcflow/u_imcflow_impl",
    "gem5": "/testbench_imcflow_gem5/u_imcflow_with_axi/u_imcflow_impl",
}


def core_path(tb: str, row: int, col: int) -> str:
    """Build the hierarchy path to a core node."""
    prefix = TB_PREFIXES[tb]
    return f"{prefix}/core_row[{row}]/core_col[{col}]"


def imce_path(tb: str, row: int, col: int) -> str:
    return f"{core_path(tb, row, col)}/imce_node/imce"


def router_path(tb: str, row: int, col: int) -> str:
    return f"{core_path(tb, row, col)}/imce_node/u_router"


def inode_path(tb: str, row: int, col: int) -> str:
    return f"{core_path(tb, row, col)}/inode/u_intf_node"


def inode_router_path(tb: str, row: int, col: int) -> str:
    return f"{core_path(tb, row, col)}/inode/u_router"


# ============================================================================
# Signal group definitions
# ============================================================================
# Each group is a function that takes (tb, row, col) and returns a list of
# (group_name, signals) tuples. Each signal is either:
#   - (full_path, None)            : addSignal with full path (sets new scope)
#   - (signal_name, None)          : addSignal -holdScope
#   - (signal_name, "UNSIGNED")    : addSignal -UNSIGNED -holdScope
#   - ("__subgroup__", name)       : addSubGroup
#   - ("__endsubgroup__", name)    : endSubGroup


def _sig(name, fmt=None):
    """Shorthand for a holdScope signal."""
    return (name, fmt)


def _fsig(path):
    """Shorthand for a full-path signal (changes scope)."""
    return (path, None)


def _sub(name):
    return ("__subgroup__", name)


def _endsub(name):
    return ("__endsubgroup__", name)


def signals_imce_ctrl(tb, row, col):
    """IMCE control pipeline signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_ctrl"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("state[1:0]", "UNSIGNED"),
        _sig("pc[7:0]"),
        _sig("compute_done"),
        _sig("compute_start"),
        _sig("recv_hs"),
        _sig("send_hs"),
        _sig("step_hs"),
        _sig("ex_stall"),
        _sig("id_stall"),
        _sig("if_stall"),
        _sig("if_flush"),
        _sig("id_bubble"),
        _sig("flag_same"),
        _sig("flag_write_enable"),
        _sig("flag_write_hs"),
        _sig("recv_fid[2:0]"),
        _sig("inst[31:0]"),
        _sig("id_inst[31:0]"),
        _sig("scan_done"),
        # compute_if
        _sub("compute_if"),
        _fsig(f"{base}/compute_if/valid"),
        _sig("ready"),
        _sig("data[63:0]"),
        _endsub("compute_if"),
        # ctrl_ex
        _sub("ctrl_ex"),
        _fsig(f"{base}/ctrl_ex/opcode[5:0]"),
        _sig("pc[7:0]"),
        _sig("rd[5:0]"),
        _sig("rs1[5:0]"),
        _sig("we"),
        _sig("fifo_id[5:0]"),
        _sig("shift_amt[2:0]"),
        _sig("blk_strobe[3:0]"),
        _sig("dwresult_valid"),
        _sig("bshr_sel[1:0]"),
        _sig("ksel"),
        _sig("is_imm"),
        _sig("imm1[13:0]"),
        _sig("qreg_we"),
        _sig("qreg_start_idx[3:0]"),
        _sig("fifo2rf"),
        _sig("fifo2lbuf"),
        _sig("layer_update"),
        _endsub("ctrl_ex"),
        # ctrl_id
        _sub("ctrl_id"),
        _fsig(f"{base}/ctrl_id/opcode[5:0]"),
        _sig("rs1[5:0]"),
        _sig("rs2[5:0]"),
        _sig("re1"),
        _sig("re2"),
        _sig("is_branch_taken"),
        _sig("is_jump"),
        _sig("inst_type[3:0]", "UNSIGNED"),
        _endsub("ctrl_id"),
        # recv_if
        _sub("recv_if"),
        _fsig(f"{base}/recv_if/valid"),
        _sig("ready"),
        _sig("data"),
        _endsub("recv_if"),
        # send_if
        _sub("send_if"),
        _fsig(f"{base}/send_if/valid"),
        _sig("ready"),
        _sig("data"),
        _endsub("send_if"),
    ]
    return [("imce_ctrl", signals)]


def signals_hazard_detector(tb, row, col):
    """Hazard detector signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_ctrl/u_hazard_detector"
    signals = [
        _fsig(f"{base}/ex_stall"),
        _sig("id_stall"),
        _sig("if_stall"),
        _sig("if_flush"),
        _sig("recv_hs"),
        _sig("send_hs"),
        _sig("step_hs"),
        _sig("flag_same"),
        _sig("flag_write_hs"),
        _sig("scan_done"),
        _sig("state[1:0]", "UNSIGNED"),
        _sub("ctrl_ex"),
        _fsig(f"{base}/ctrl_ex/opcode[5:0]"),
        _sig("rd[5:0]"),
        _sig("we"),
        _sig("fifo_id[5:0]"),
        _endsub("ctrl_ex"),
    ]
    return [("hazard_detector", signals)]


def signals_ctrl_pl(tb, row, col):
    """Control pipeline signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_ctrl/u_ctrl_pl"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("state[1:0]", "UNSIGNED"),
        _sig("ex_stall"),
        _sig("id_stall"),
        _sig("inst_type[3:0]", "UNSIGNED"),
        _sig("recv_valid"),
        _sig("recv_cmd[4:0]", "UNSIGNED"),
        _sig("recv_addr[15:0]"),
        _sig("ex_we"),
        _sig("ex_rd[5:0]"),
        _sub("ctrl_ex"),
        _fsig(f"{base}/ctrl_ex/opcode[5:0]"),
        _sig("rd[5:0]"),
        _sig("rs1[5:0]"),
        _sig("we"),
        _sig("fifo_id[5:0]"),
        _sig("shift_amt[2:0]"),
        _sig("blk_strobe[3:0]"),
        _sig("dwresult_valid"),
        _sig("bshr_sel[1:0]"),
        _sig("ksel"),
        _sig("imm1[13:0]"),
        _sig("imm2[13:0]"),
        _sig("is_imm"),
        _sig("qreg_start_idx[3:0]"),
        _sig("qreg_we"),
        _sig("recv_rd_is_zero"),
        _sig("recv_addr_is_zero"),
        _sig("layer_update"),
        _sig("fifo2rf"),
        _sig("fifo2lbuf"),
        _sig("is_cmd_rf_write"),
        _sig("mem_en"),
        _sig("mem_we"),
        _endsub("ctrl_ex"),
        _sub("ctrl_id"),
        _fsig(f"{base}/ctrl_id/opcode[5:0]"),
        _sig("inst_type[3:0]", "UNSIGNED"),
        _sig("rs1[5:0]"),
        _sig("rs2[5:0]"),
        _sig("re1"),
        _sig("re2"),
        _sig("imm1[13:0]"),
        _sig("imm2[13:0]"),
        _sig("imm3[13:0]"),
        _sig("fifo_id[5:0]"),
        _sig("flag_value"),
        _sig("is_branch_taken"),
        _sig("is_jump"),
        _endsub("ctrl_id"),
    ]
    return [("ctrl_pl", signals)]


def signals_datapath(tb, row, col):
    """IMCE datapath top-level signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_datapath"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("state[1:0]", "UNSIGNED"),
        _sig("rdata_b_o[255:0]"),
        _sig("scan_done"),
        _sub("ctrl_ex"),
        _fsig(f"{base}/ctrl_ex/opcode[5:0]"),
        _sig("rd[5:0]"),
        _sig("we"),
        _sig("shift_amt[2:0]"),
        _sig("dwresult_valid"),
        _sig("bshr_sel[1:0]"),
        _sig("qreg_we"),
        _endsub("ctrl_ex"),
        _sub("bshr_set_if"),
        _fsig(f"{base}/bshr_set/valid"),
        _sig("ready"),
        _endsub("bshr_set_if"),
        _sub("pimc_if"),
        _fsig(f"{base}/pimc/valid"),
        _sig("ready"),
        _sig("data[63:0]"),
        _endsub("pimc_if"),
    ]
    return [("imce_datapath", signals)]


def signals_erf(tb, row, col):
    """Extended register file signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_erf"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("gpr_we"),
        _sig("gpr_waddr"),
        _sig("gpr_wdata[255:0]"),
        _sig("gpr_re_a"),
        _sig("gpr_re_b"),
        _sig("rdata_a_o[255:0]"),
        _sig("rdata_b_o[255:0]"),
        _sig("read_a_from_gpr"),
        _sig("read_b_from_gpr"),
        _sig("fifo2rf_hs"),
    ]
    return [("erf", signals)]


def signals_vpu(tb, row, col):
    """VPU signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_vpu"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("opcode_i[5:0]", "UNSIGNED"),
        _sig("err_o[15:0]"),
        _sig("block_mask_o[15:0]"),
        _sig("src_mask[3:0]"),
        _sig("shift_amt[2:0]"),
        _sig("dwresult_valid"),
        _sig("min[15:0]"),
        _sig("max[15:0]"),
        _sig("ksel_i"),
        _sig("bshr_sel[1:0]"),
    ]
    return [("vpu", signals)]


def signals_linebuffer(tb, row, col):
    """Linebuffer signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_linebuffer"
    ctrl = f"{base}/ctrl"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sub("input_rx"),
        _fsig(f"{base}/input_rx/valid"),
        _sig("ready"),
        _sig("data[255:0]"),
        _endsub("input_rx"),
        _sub("bshr_tx"),
        _fsig(f"{base}/bshr_tx/valid"),
        _sig("ready"),
        _endsub("bshr_tx"),
        _sub("bshr_set_tx"),
        _fsig(f"{base}/bshr_set_tx/valid"),
        _sig("ready"),
        _endsub("bshr_set_tx"),
        # addr_shfl_gen — signals for debugging in_ready_o
        _sub("ready_debug"),
        _fsig(f"{ctrl}/in_ready_o"),
        _sig("pipeline_filled"),
        _sig("S0_lbuf_filled"),
        _sig("S0_bshr_filled"),
        _sig("S0_is_right_pad"),
        _sig("S0_is_bottom_pad"),
        _sig("S0_out_valid"),
        _sig("S1_out_valid"),
        _sig("S2_out_valid"),
        _sig("S3_out_valid"),
        _sig("S3_ready_i"),
        _sig("S0_in_transfer"),
        _sig("S0_right_pad_transfer"),
        _sig("S0_bottom_pad_transfer"),
        _sig("S0_in_pad_transfer"),
        _sig("all_recived"),
        _sig("S0_row[9:0]", "UNSIGNED"),
        _sig("S0_col[9:0]", "UNSIGNED"),
        _sig("S0_bitpos[1:0]", "UNSIGNED"),
        _sig("S3_bitpos[1:0]", "UNSIGNED"),
        _endsub("ready_debug"),
    ]
    return [("linebuffer", signals)]


def signals_imcu(tb, row, col):
    """IMCU core top-level signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_imcu_core"
    signals = [
        _sub("core_rx"),
        _fsig(f"{base}/core_rx/valid"),
        _sig("ready"),
        _sig("data[255:0]"),
        _endsub("core_rx"),
        _sub("core_tx"),
        _fsig(f"{base}/core_tx/valid"),
        _sig("ready"),
        _sig("data[63:0]"),
        _endsub("core_tx"),
    ]
    return [("imcu_core", signals)]


def signals_imcu_unit(tb, row, col):
    """IMCU compute unit signals (imcu.sv).
    Hierarchy: u_imce_datapath/u_imcu_core/u_imcu
    """
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_imcu_core/u_imcu"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rst_i"),
        _sig("sram_c_en_i"),
        _sig("cim_c_en_i"),
        _sig("valid_o"),
        _sig("phase_cnt"),
        _sig("adc_ready"),
        _sig("adc_ready_q"),
        _sig("in_phase"),
        _sig("phase_end"),
        _sig("cnt[2:0]"),
        _sig("bitpos[2:0]"),
        _sig("cnt_q[2:0]"),
    ]
    return [("imcu_unit", signals)]


def signals_post_imcu(tb, row, col):
    """Post-IMCU processing signals (post_imcu.sv).
    Hierarchy: u_imce_datapath/u_imcu_core/u_post_imcu
    """
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_imcu_core/u_post_imcu"
    serial = f"{base}/u_serializer"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("en_i"),
        _sig("imcu_valid_i"),
        _sig("pimc_valid"),
        _sub("u_serializer"),
        _fsig(f"{serial}/valid_i"),
        _fsig(f"{serial}/valid_o"),
        _sig("v_cnt[2:0]"),
        _sig("v_cnt_q[2:0]"),
        _endsub("u_serializer"),
    ]
    return [("post_imcu", signals)]


def signals_imcu_ctrl(tb, row, col):
    """IMCU controller signals."""
    base = f"{imce_path(tb, row, col)}/u_imce_datapath/u_imcu_core/u_imcu_ctrl"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sub("lbuf_ready_part"),
        _sub("core_rx"),
        _fsig(f"{base}/core_rx/valid"),
        _fsig(f"{base}/core_rx/ready"),
        _endsub("core_rx"),
        _sub("core_tx"),
        _fsig(f"{base}/core_tx/valid"),
        _fsig(f"{base}/core_tx/ready"),
        _endsub("core_tx"),
        _fsig(f"{base}/en_i"),
        _sig("is_imcu_mode"),
        _sig("cim_c_en_o"),
        _sig("core_ready"),
        _sig("cim_cnt"),
        _endsub("lbuf_ready_part"),
    ]
    return [("imcu_ctrl", signals)]


def signals_router(tb, row, col, is_inode=False):
    """Router + arbiter signals."""
    base = router_path(tb, row, col) if not is_inode else inode_router_path(tb, row, col)
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sub("u_arbiter"),
        _fsig(f"{base}/u_arbiter/gnt_o[4:0]"),
        _sig("po_req[4:0]"),
        _sig("po_gnt[4:0]"),
        _sig("po_transfer[4:0]"),
        _sig("po_queue[4:0]"),
        _sig("po_remain[4:0]"),
        _sig("hs[4:0]"),
        _endsub("u_arbiter"),
        _sub("u_input_block"),
        _fsig(f"{base}/u_input_block/policy_cmd"),
        _endsub("u_input_block"),
    ]
    return [("router", signals)]


# ============================================================================
# Inode signal groups
# ============================================================================

def signals_inode_fsm(tb, row, col):
    """Inode FSM signals (intf_node_fsm.sv).
    Hierarchy: inode/u_intf_node/u_intf_node_fsm
    """
    base = f"{inode_path(tb, row, col)}/u_intf_node_fsm"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("fsm_state"),
        _sig("pc_reg_i[31:0]"),
        _sig("run_req_i"),
        _sig("halt_i"),
        _sig("run_ack_o"),
        _sig("start_o"),
        _sig("inode_state_o"),
        _sig("start_pc_o[31:0]"),
        _sig("is_active_o"),
    ]
    return [("inode_fsm", signals)]


def signals_inode_if_stage(tb, row, col):
    """Inode IF stage signals (IF_stage.sv).
    Hierarchy: inode/u_intf_node/if_stage
    Submodules: u_pc_gen_intf_node, u_imem_intf_node
    """
    base = f"{inode_path(tb, row, col)}/if_stage"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("start_i"),
        _sig("start_pc_i[31:0]"),
        _sig("is_active_i"),
        _sig("pc_sel_i"),
        _sig("stall_i"),
        _sig("flush_i"),
        _sig("inst_o[31:0]"),
        _sig("pc_o[7:0]"),
        _sig("last_pc[31:0]"),
        _sig("flush_q"),
    ]
    return [("inode_if_stage", signals)]


def signals_inode_id_stage(tb, row, col):
    """Inode ID stage signals (ID_stage.sv).
    Hierarchy: inode/u_intf_node/id_stage
    Submodules: u_decoder_intf_node, u_reg_file_intf_node, u_ctrl_generator
    """
    base = f"{inode_path(tb, row, col)}/id_stage"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("inst_i[31:0]"),
        _sig("opcode_o"),
        _sig("rs1_o"),
        _sig("rs2_o"),
        _sig("rd_o"),
        _sig("imm_o"),
        _sig("dmem_ren_o"),
        _sig("dmem_wen_o"),
        _sig("packet_en_o"),
        _sig("srf_wen_o"),
        _sig("srf_of_rs1_o"),
        _sig("srf_of_rs2_o"),
        _sig("fifo_id_o"),
        _sig("node_col_id_o"),
        _sig("interrupt_id_o"),
        _sig("pc_br_target_o[7:0]"),
        _sig("pc_p4_o[7:0]"),
        _sig("is_flush"),
        _sig("rs1_forward_i"),
        _sig("rs2_forward_i"),
    ]
    return [("inode_id_stage", signals)]


def signals_inode_ex_stage(tb, row, col):
    """Inode EX stage signals (EX_stage.sv).
    Hierarchy: inode/u_intf_node/ex_stage
    Submodules: u_recv_fifo
    """
    base = f"{inode_path(tb, row, col)}/ex_stage"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("opcode_i"),
        _sig("operand1_sel_i[1:0]"),
        _sig("operand2_sel_i[1:0]"),
        _sig("srf_of_rs1_i"),
        _sig("srf_of_rs2_i"),
        _sig("imm_i"),
        _sig("no_true_dep_i"),
        _sig("alu_result_o"),
        _sig("is_branch_taken_o"),
        _sig("pc_i[7:0]"),
        _sig("pc_p4_o[7:0]"),
        _sig("forwarded_srf_of_rs2_o"),
        _sub("recv_fifo"),
        _sig("recv_fifo_pop_valid_o"),
        _sig("recv_fifo_pop_ready_o"),
        _sig("recv_fifo_data_o"),
        _endsub("recv_fifo"),
        _sub("sync_reg"),
        _sig("syn_reg_req_o"),
        _sig("sync_reg_req_data_o"),
        _endsub("sync_reg"),
    ]
    return [("inode_ex_stage", signals)]


def signals_inode_mem_stage(tb, row, col):
    """Inode MEM stage signals (MEM_stage.sv).
    Hierarchy: inode/u_intf_node/mem_stage
    Submodules: u_mem (tc_sram)
    """
    base = f"{inode_path(tb, row, col)}/mem_stage"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("opcode_i"),
        _sig("dmem_ren_i"),
        _sig("dmem_wen_i"),
        _sig("addr_i"),
        _sig("data_i"),
        _sig("reg_data_i"),
        _sig("is_active_i"),
        _sig("data_o"),
        _sig("reg_data_o"),
        _sub("sram_internal"),
        _sig("csn"),
        _sig("we"),
        _sig("addr"),
        _sig("wdata"),
        _sig("rdata"),
        _endsub("sram_internal"),
    ]
    return [("inode_mem_stage", signals)]


def signals_inode_wb_stage(tb, row, col):
    """Inode WB stage signals (WB_stage.sv).
    Hierarchy: inode/u_intf_node/wb_stage
    Submodules: u_packet_packer, u_send_fifo_block
    """
    base = f"{inode_path(tb, row, col)}/wb_stage"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("opcode_i"),
        _sig("packet_en_i"),
        _sig("alu_result_i"),
        _sig("data_i"),
        _sig("reg_data_i"),
        _sig("halt_o"),
        _sig("interrupt_valid_o"),
        _sig("interrupt_id_o"),
        _sig("rd_reg_data_o"),
        _sub("packet_tx"),
        _sig("packet_tx_req_o"),
        _sig("packet_tx_gnt_o"),
        _sig("node_col_id_i"),
        _sig("next_policy_addr_i"),
        _sig("imm_i"),
        _sig("fifo_id_i"),
        _sig("forwarded_rs2_i"),
        _endsub("packet_tx"),
    ]
    return [("inode_wb_stage", signals)]


def signals_inode_hazard(tb, row, col):
    """Inode hazard + forward control signals.
    Hierarchy: inode/u_intf_node/u_hazard_control, u_forward_control
    """
    base = f"{inode_path(tb, row, col)}"
    hzd = f"{base}/u_hazard_control"
    fwd = f"{base}/u_forward_control"
    signals = [
        _sub("hazard_control"),
        _fsig(f"{hzd}/clk_i"),
        _sig("rstn_i"),
        _sig("is_active_i"),
        _sig("start_i"),
        _sig("if_pc_sel_o"),
        _sig("if_stall_o"),
        _sig("if_flush_o"),
        _sig("id_opcode_i"),
        _sig("id_stall_o"),
        _sig("id_flush_o"),
        _sig("ex_opcode_i"),
        _sig("ex_no_true_dep_o"),
        _sig("ex_stall_o"),
        _sig("ex_flush_o"),
        _sig("ex_recv_fifo_req_i"),
        _sig("ex_recv_fifo_gnt_i"),
        _sig("ex_branch_not_taken_i"),
        _sig("mem_stall_o"),
        _sig("mem_flush_o"),
        _sig("wb_packet_en_i"),
        _sig("wb_send_fifo_req_i"),
        _sig("wb_send_fifo_gnt_i"),
        _sig("wb_interrupt_req_i"),
        _sig("wb_interrupt_ack_i"),
        _sig("wb_stall_o"),
        _sig("wb_flush_o"),
        _sig("id_is_br"),
        _sig("ex_is_br"),
        _sig("id_is_halt"),
        _endsub("hazard_control"),
        _sub("forward_control"),
        _fsig(f"{fwd}/clk_i"),
        _sig("rstn_i"),
        _sig("id_rs1_i"),
        _sig("id_rs2_i"),
        _sig("id_rs1_forward_o"),
        _sig("id_rs2_forward_o"),
        _sig("ex_rs1_i"),
        _sig("ex_rs2_i"),
        _sig("mem_rd_i"),
        _sig("mem_reg_we_i"),
        _sig("wb_rd_i"),
        _sig("wb_reg_we_i"),
        _sig("operand1_sel_o"),
        _sig("operand2_sel_o"),
        _endsub("forward_control"),
    ]
    return [("inode_hazard", signals)]


def signals_fifo_block(tb, row, col):
    """FIFO block signals."""
    base = f"{imce_path(tb, row, col)}/u_fifo_block"
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("pop_id[2:0]"),
        _sub("pop_tx"),
        _fsig(f"{base}/pop_tx/valid"),
        _sig("ready"),
        _sig("data"),
        _endsub("pop_tx"),
    ]
    return [("fifo_block", signals)]


# ============================================================================
# Cross-module signal groups (grouped by semantic meaning, not by module)
# ============================================================================

def signals_op_step(tb, row, col):
    """OP_STEP related signals across modules."""
    imcu_ctrl = f"{imce_path(tb, row, col)}/u_imce_datapath/u_imcu_core/u_imcu_ctrl"
    imce_ctrl = f"{imce_path(tb, row, col)}/u_imce_ctrl"
    hazard = f"{imce_ctrl}/u_hazard_detector"
    lbuf = f"{imce_path(tb, row, col)}/u_imce_datapath/u_linebuffer"
    lbuf_ctrl = f"{lbuf}/ctrl"
    dp = f"{imce_path(tb, row, col)}/u_imce_datapath"
    signals = [
        _sub("imcu_ctrl"),
        _fsig(f"{imcu_ctrl}/clk_i"),
        _sub("core_rx"),
        _fsig(f"{imcu_ctrl}/core_rx/valid"),
        _fsig(f"{imcu_ctrl}/core_rx/ready"),
        _endsub("core_rx"),
        _sub("core_tx"),
        _fsig(f"{imcu_ctrl}/core_tx/valid"),
        _fsig(f"{imcu_ctrl}/core_tx/ready"),
        _endsub("core_tx"),
        _fsig(f"{imcu_ctrl}/en_i"),
        _sig("is_imcu_mode"),
        _sig("cim_c_en_o"),
        _sig("core_ready"),
        _sig("cim_cnt"),
        _endsub("imcu_ctrl"),

        _sub("imce_ctrl"),
        _fsig(f"{imce_ctrl}/clk_i"),
        _sig("state[1:0]", "UNSIGNED"),
        _sig("pc[7:0]"),
        _sig("step_hs"),
        _sig("ex_stall"),
        _sig("id_stall"),
        _sig("if_stall"),
        # compute_if
        _sub("compute_if"),
        _fsig(f"{imce_ctrl}/compute_if/valid"),
        _sig("ready"),
        _endsub("compute_if"),
        _sub("ctrl_ex"),
        _fsig(f"{imce_ctrl}/ctrl_ex/opcode[5:0]"),
        _sig("layer_update"),
        _endsub("ctrl_ex"),
        _endsub("imce_ctrl"),

        _sub("lbuf"),
        _fsig(f"{lbuf}/clk_i"),
        _sub("input_rx"),
        _fsig(f"{lbuf}/input_rx/valid"),
        _sig("ready"),
        _sig("data[255:0]"),
        _endsub("input_rx"),
        _sub("bshr_tx"),
        _fsig(f"{lbuf}/bshr_tx/valid"),
        _sig("ready"),
        _endsub("bshr_tx"),
        _sub("ready_debug"),
        _fsig(f"{lbuf_ctrl}/in_ready_o"),
        _sig("pipeline_filled"),
        _sig("S0_lbuf_filled"),
        _sig("S0_bshr_filled"),
        _sig("S0_is_right_pad"),
        _sig("S0_is_bottom_pad"),
        _sig("S0_out_valid"),
        _sig("S1_out_valid"),
        _sig("S2_out_valid"),
        _sig("S3_out_valid"),
        _sig("S3_ready_i"),
        _sig("S0_in_transfer"),
        _sig("S0_right_pad_transfer"),
        _sig("S0_bottom_pad_transfer"),
        _sig("S0_in_pad_transfer"),
        _sig("all_recived"),
        _sig("S0_row[9:0]", "UNSIGNED"),
        _sig("S0_col[9:0]", "UNSIGNED"),
        _sig("S0_bitpos[1:0]", "UNSIGNED"),
        _sig("S3_bitpos[1:0]", "UNSIGNED"),
        _endsub("ready_debug"),
        _endsub("lbuf"),
    ]
    return [("op_step", signals)]


# ============================================================================
# Top-level signal groups (not per-node, generated once at the impl level)
# ============================================================================

def signals_imcflow_impl(tb):
    """IMCFlow impl top-level signals."""
    base = TB_PREFIXES[tb]
    signals = [
        _fsig(f"{base}/clk_i"),
        _sig("rstn_i"),
        _sig("io_req_i"),
        _sig("io_gnt_o"),
        _sig("io_addr_i"),
        _sig("io_wen_i"),
        _sig("io_be_i"),
        _sig("io_data_i"),
        _sig("io_r_data_o"),
        _sig("io_r_valid_o"),
    ]
    return [("imcflow_impl", signals)]


# Top-level groups: function takes (tb) only, not (tb, row, col)
TOP_LEVEL_GROUPS: Dict[str, callable] = {
    "impl": signals_imcflow_impl,
}

# Which top-level groups to include by default
DEFAULT_TOP_LEVEL_GROUPS = ["impl"]


# Registry of all available signal groups (per-node)
SIGNAL_GROUPS: Dict[str, callable] = {
    "ctrl":      signals_imce_ctrl,
    "hazard":    signals_hazard_detector,
    "ctrl_pl":   signals_ctrl_pl,
    "datapath":  signals_datapath,
    "erf":       signals_erf,
    "vpu":       signals_vpu,
    "linebuffer": signals_linebuffer,
    "imcu":       signals_imcu,
    "imcu_unit":  signals_imcu_unit,
    "post_imcu":  signals_post_imcu,
    "imcu_ctrl":  signals_imcu_ctrl,
    "router":    signals_router,
    "fifo":      signals_fifo_block,
    # cross-module groups
    "op_step":   signals_op_step,
    # inode groups
    "inode_fsm":       signals_inode_fsm,
    "inode_if":        signals_inode_if_stage,
    "inode_id":        signals_inode_id_stage,
    "inode_ex":        signals_inode_ex_stage,
    "inode_mem":       signals_inode_mem_stage,
    "inode_wb":        signals_inode_wb_stage,
    "inode_hazard":    signals_inode_hazard,
}

# Default groups for IMCE nodes
DEFAULT_IMCE_GROUPS = ["ctrl", "hazard", "datapath", "vpu", "linebuffer", "erf", "imcu", "imcu_unit", "post_imcu", "imcu_ctrl", "router", "fifo", "op_step"]

# Default groups for inode (col=0)
DEFAULT_INODE_GROUPS = ["router", "inode_fsm", "inode_if", "inode_id", "inode_ex", "inode_mem", "inode_wb", "inode_hazard"]


# ============================================================================
# RC file generation
# ============================================================================

ROW_COLORS = [
    "ID_BLUE4", "ID_GREEN4", "ID_YELLOW5", "ID_RED4",
    "ID_CYAN4", "ID_MAGENTA4", "ID_ORANGE4", "ID_WHITE",
]


def format_signal_line(sig_tuple, height=15):
    """Format a single signal entry for the .rc file."""
    name, fmt = sig_tuple

    if name == "__subgroup__":
        return f'addSubGroup "{fmt}" -e FALSE'
    if name == "__endsubgroup__":
        return f'endSubGroup "{fmt}"'

    # Full path signal (starts with /)
    if name.startswith("/"):
        if fmt == "UNSIGNED":
            return f"addSignal -h {height} -UNSIGNED {name}"
        return f"addSignal -h {height} {name}"

    # holdScope signal
    if fmt == "UNSIGNED":
        return f"addSignal -h {height} -UNSIGNED -holdScope {name}"
    return f"addSignal -h {height} -holdScope {name}"


def generate_rc(
    fsdb_path: str,
    nodes: List[Tuple[int, int]],
    tb: str = "gem5",
    groups: Optional[List[str]] = None,
    collapsed: bool = True,
) -> str:
    """Generate the full .rc file content."""

    lines = []

    # Header
    lines.append("Magic 271485")
    lines.append("Revision Verdi_R-2020.12-SP1")
    lines.append("")
    lines.append("; Window Layout <x> <y> <width> <height> <signalwidth> <valuewidth>")
    lines.append("viewPort 0 25 2560 1250 350 330")
    lines.append("")
    lines.append("; File list:")
    lines.append(f'openDirFile -d / "" "{fsdb_path}"')
    lines.append("")
    lines.append("; signal spacing:")
    lines.append("signalSpacing 5")
    lines.append("")
    lines.append("; waveform viewport range")
    lines.append("zoom 0.000000 1000000.000000")
    lines.append("cursor 0.000000")
    lines.append("marker 0.000000")
    lines.append("")
    lines.append("COMPLEX_EVENT_BEGIN")
    lines.append("COMPLEX_EVENT_END")
    lines.append("")
    lines.append("curSTATUS ByChange")
    lines.append("")

    # Active file
    lines.append(f'activeDirFile "" "{fsdb_path}"')
    lines.append("")

    # Generate top-level signal groups (once, not per-node)
    top_groups = [g for g in (groups or DEFAULT_TOP_LEVEL_GROUPS) if g in TOP_LEVEL_GROUPS]
    for grp_name in top_groups:
        gen_fn = TOP_LEVEL_GROUPS[grp_name]
        group_entries = gen_fn(tb)
        for sub_name, signals in group_entries:
            collapse_flag = " -e FALSE" if collapsed else ""
            lines.append(f'addGroup "{sub_name}" -c ID_WHITE{collapse_flag}')
            for sig in signals:
                lines.append(format_signal_line(sig))
        lines.append("")

    # Generate signal groups for each node, sorted by (row, col)
    for row, col in sorted(nodes):
        is_inode = (col == 0)
        node_label = f"inode({row},{col})" if is_inode else f"imce({row},{col})"

        # Determine which groups to use
        if groups:
            node_groups = groups
        else:
            node_groups = DEFAULT_INODE_GROUPS if is_inode else DEFAULT_IMCE_GROUPS

        color = ROW_COLORS[row % len(ROW_COLORS)]
        collapse_flag = " -e FALSE" if collapsed else ""
        lines.append(f'addGroup "{node_label}" -c {color}{collapse_flag}')

        for grp_name in node_groups:
            if grp_name not in SIGNAL_GROUPS:
                continue

            gen_fn = SIGNAL_GROUPS[grp_name]

            # Router has an is_inode parameter
            if grp_name == "router":
                group_entries = gen_fn(tb, row, col, is_inode=is_inode)
            else:
                group_entries = gen_fn(tb, row, col)

            # Wrap each signal group as a subGroup inside the node group
            for sub_name, signals in group_entries:
                lines.append(f'addSubGroup "{sub_name}" -e FALSE')
                for sig in signals:
                    lines.append(format_signal_line(sig))
                lines.append(f'endSubGroup "{sub_name}"')

        lines.append("")

    # Scope hierarchy section
    lines.append("")
    lines.append("GETSIGNALFORM_SCOPE_HIERARCHY_BEGIN")
    lines.append('getSignalForm close')
    lines.append("GETSIGNALFORM_SCOPE_HIERARCHY_END")
    lines.append("")
    lines.append("FILTER_SIGNAL_BEGIN")
    lines.append('""')
    lines.append("FILTER_STRING_LIST_BEGIN")
    lines.append("FILTER_STRING_LIST_END")
    lines.append("FILTER_TYPE_LIST_BEGIN")
    lines.append('"All"')
    lines.append('"Input"')
    lines.append('"Output"')
    lines.append('"Inout"')
    lines.append('"Net"')
    lines.append('"Register"')
    lines.append("FILTER_TYPE_LIST_END")
    lines.append("FILTER_SIGNAL_END")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Node discovery
# ============================================================================

def parse_nodes_from_hw_node_map(test_dir: str) -> List[Tuple[int, int]]:
    """Extract unique (row, col) from hw_node_map.txt."""
    map_path = os.path.join(test_dir, "hw_node_map.txt")
    coords = set()
    pattern = re.compile(r'(?:imce|inode)_(\d+)_(\d+)')

    with open(map_path) as f:
        for match in pattern.finditer(f.read()):
            coords.add((int(match.group(1)), int(match.group(2))))

    return sorted(coords)


def parse_nodes_from_fsim_logs(test_dir: str) -> List[Tuple[int, int]]:
    """Extract unique (row, col) from fsim_logs directory."""
    fsim_dir = os.path.join(test_dir, "logs", "rtl_runner", "fsim_logs")
    coords = set()
    pattern = re.compile(r'core_row_(\d+)_\.core_col_(\d+)_')

    if os.path.isdir(fsim_dir):
        for fname in os.listdir(fsim_dir):
            match = pattern.search(fname)
            if match:
                coords.add((int(match.group(1)), int(match.group(2))))

    return sorted(coords)


def find_fsdb(test_dir: str) -> Optional[str]:
    """Find the FSDB file in a test directory."""
    rtl_dir = os.path.join(test_dir, "logs", "rtl_runner")
    if os.path.isdir(rtl_dir):
        for fname in os.listdir(rtl_dir):
            if fname.endswith(".fsdb"):
                return os.path.join(rtl_dir, fname)
    return None


def grid_nodes(grid_str: str) -> List[Tuple[int, int]]:
    """Parse 'RxC' grid string into list of all (row, col) coordinates."""
    rows, cols = map(int, grid_str.lower().split("x"))
    return [(r, c) for r in range(rows) for c in range(cols)]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Verdi .rc signal preference file for IMCFlow RTL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specific nodes with FSDB path
  python gen_verdi_rc.py --nodes 0,1 1,2 3,3 --fsdb sim.fsdb -o signals.rc

  # All nodes from test directory (auto-discovers nodes and FSDB)
  python gen_verdi_rc.py --test-dir ../ds_cnn_full_pretrained_evl -o signals.rc

  # Full 4x5 grid
  python gen_verdi_rc.py --grid 4x5 --fsdb sim.fsdb -o signals.rc

  # Only router and ctrl groups
  python gen_verdi_rc.py --nodes 0,1 --groups ctrl router --fsdb sim.fsdb -o signals.rc

  # Standalone testbench (not gem5)
  python gen_verdi_rc.py --nodes 0,1 --tb standalone --fsdb inter.fsdb -o signals.rc

Available signal groups: """ + ", ".join(sorted({**SIGNAL_GROUPS, **TOP_LEVEL_GROUPS}.keys()))
    )

    parser.add_argument("-o", "--output", required=True, help="Output .rc file path")
    parser.add_argument("--fsdb", help="Path to FSDB file (auto-detected with --test-dir)")
    parser.add_argument("--tb", choices=["gem5", "standalone"], default="gem5",
                        help="Testbench type (default: gem5)")

    # Node selection (mutually exclusive)
    node_group = parser.add_mutually_exclusive_group(required=True)
    node_group.add_argument("--nodes", nargs="+", metavar="R,C",
                            help="List of row,col coordinates (e.g., 0,1 1,2 3,3)")
    node_group.add_argument("--test-dir", help="Test directory (auto-discovers nodes from hw_node_map.txt)")
    node_group.add_argument("--grid", metavar="RxC",
                            help="Generate for full RxC grid (e.g., 4x5)")

    all_groups = sorted({**SIGNAL_GROUPS, **TOP_LEVEL_GROUPS}.keys())
    parser.add_argument("--groups", nargs="+", choices=all_groups,
                        help="Signal groups to include (default: all)")
    parser.add_argument("--expanded", action="store_true",
                        help="Start groups expanded (default: collapsed)")

    args = parser.parse_args()

    # Resolve nodes
    if args.nodes:
        nodes = []
        for n in args.nodes:
            r, c = map(int, n.split(","))
            nodes.append((r, c))
    elif args.test_dir:
        try:
            nodes = parse_nodes_from_hw_node_map(args.test_dir)
        except FileNotFoundError:
            nodes = parse_nodes_from_fsim_logs(args.test_dir)
        if not nodes:
            print(f"Error: no nodes found in {args.test_dir}")
            return 1
        print(f"Discovered {len(nodes)} nodes: {nodes}")
    elif args.grid:
        nodes = grid_nodes(args.grid)
    else:
        print("Error: specify --nodes, --test-dir, or --grid")
        return 1

    # Resolve FSDB path
    fsdb_path = args.fsdb
    if not fsdb_path and args.test_dir:
        fsdb_path = find_fsdb(args.test_dir)
    if not fsdb_path:
        fsdb_path = "REPLACE_WITH_FSDB_PATH.fsdb"
        print(f"Warning: no FSDB file specified, using placeholder: {fsdb_path}")

    # Generate
    content = generate_rc(
        fsdb_path=fsdb_path,
        nodes=nodes,
        tb=args.tb,
        groups=args.groups,
        collapsed=not args.expanded,
    )

    with open(args.output, "w") as f:
        f.write(content)

    print(f"Generated {args.output} with {len(nodes)} nodes")
    print(f"  FSDB: {fsdb_path}")
    print(f"  TB: {args.tb}")
    print(f"  Groups: {args.groups or 'all defaults'}")
    return 0


if __name__ == "__main__":
    exit(main())
