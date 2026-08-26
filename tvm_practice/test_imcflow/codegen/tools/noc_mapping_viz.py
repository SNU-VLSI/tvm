#!/usr/bin/env python3
"""noc_mapping_viz.py <eval_dir>

Self-contained NoC mapping visualizer. Re-derives, DIRECTLY from an eval_dir's
debug artifacts, (a) the region -> IMCE -> NN-layer mapping and (b) the XY-routed
per-link NoC channel occupancy, then renders two figure styles into
<eval_dir>/noc_visualizations/:
  region_layer_map_<model>.{png,svg}   region grid, same NN layer = same color
  noc_mesh_<model>.{png,svg}           4x5 mesh (INODE|IMCEx4) + per-link channels
  region_layer_map.json                extracted mapping
  noc_link_channels.json               XY-routed link-channel occupancy

Inputs (read from <eval_dir>): active_imce_list.txt, final_imcflow_config_HWNodeMap.txt,
custom_id_to_name.txt, 15_with_mappings.txt, split_info.txt, noc_paths.txt.

Routing: XY dimension-order (column/horizontal first, then row/vertical), coords
(row,col)=divmod(NodeID.value,5) [tvm/contrib/imcflow.py:472, NODE_COL_NUM=5].
This reproduces joint_pnr_results Total-hops & Routes exactly for resnet8+ds_cnn.

Layer naming degrades gracefully: known families (resnet8, ds_cnn/kws) get human
layer names (conv1,b1.c1,pw2,...); unknown models fall back to op-name labels so
the tool never crashes on an unseen network. Guarded to never raise to the caller.
"""
import os, re, sys, json, ast

# ------------------------- artifact parsing helpers -------------------------

def _read(eval_dir, name):
    p = os.path.join(eval_dir, name)
    return open(p).read() if os.path.exists(p) else None

def parse_active_imce(eval_dir):
    """{ region_func: set('imce_r_c', ...) }"""
    txt = _read(eval_dir, "active_imce_list.txt") or ""
    out = {}
    for m in re.finditer(r"'([^']*region\d+[^']*)':\s*\[(.*?)\]", txt, re.S):
        reg = m.group(1)
        ids = re.findall(r"imce_(\d+_\d+)", m.group(2))
        out[reg] = set("imce_" + x for x in ids)
    return out

def parse_hwnodemap(eval_dir):
    """{ custom_id(int): [hw_node_short, ...] }  (a custom_id may map to >1 node)"""
    txt = _read(eval_dir, "final_imcflow_config_HWNodeMap.txt") or ""
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\s*'(-?\d+)\s*:\s*(.*)'", line.strip())
        if not m:
            continue
        cid = int(m.group(1)); rhs = m.group(2)
        nodes = re.findall(r"(imce_\d+_\d+|inode_\d+_\d+)", rhs)
        if nodes:
            out.setdefault(cid, [])
            out[cid].extend(nodes)
    return out

def parse_custom_id_to_name(eval_dir):
    txt = _read(eval_dir, "custom_id_to_name.txt") or "{}"
    try:
        return {int(k): v for k, v in ast.literal_eval(txt).items()}
    except Exception:
        d = {}
        for m in re.finditer(r"(-?\d+):\s*'([^']*)'", txt):
            d[int(m.group(1))] = m.group(2)
        return d

# op-name token -> compact tag, in relay-op granularity (used to expand a
# composite's PartitionedFromPattern into its fused chain, e.g.
# "nn.imcflow_qconv_imcflow.fused_batch_norm_qnn.imcflow_min_max_quantize_"
# -> "qconv+bn+mm" -- the mesh otherwise under-reports fusion as bare "qconv").
_PATTERN_TOKENS = [
    ("nn.imcflow_qdwconv", "dwconv"), ("nn.imcflow_qconv", "qconv"),
    ("imcflow.fused_batch_norm", "bn"), ("qnn.imcflow_min_max_quantize", "mm"),
    ("nn.relu", "relu"), ("multiply", "mul"), ("add", "add"),
]

def _pattern_chain(pat):
    toks, rest = [], pat
    while rest:
        for key, tag in _PATTERN_TOKENS:
            if rest.startswith(key + "_") or rest == key or rest.startswith(key):
                toks.append(tag)
                rest = rest[len(key):].lstrip("_")
                break
        else:
            rest = rest[1:]  # skip unrecognized char (defensive)
    return toks

def parse_composite_patterns(eval_dir):
    """{composite custom_id: [fused op tags]} from 15_with_mappings.txt --
    each composite fn line carries PartitionedFromPattern= and custom_id=."""
    txt = _read(eval_dir, "15_with_mappings.txt") or ""
    out = {}
    for line in txt.splitlines():
        pm = re.search(r'PartitionedFromPattern="([^"]+)"', line)
        cm = re.search(r"custom_id=(\d+)", line)
        if pm and cm:
            out[int(cm.group(1))] = _pattern_chain(pm.group(1))
    return out

def parse_impl_ops(eval_dir):
    """{region_func: {node: [op tags]}} from the GENERATED build imce.cpp --
    the definitive fused-chain ground truth (the '// generate:' markers inside
    each hid/wid section). Avoids any custom-id guessing for composites."""
    import glob as _glob
    out = {}
    for cpp in _glob.glob(os.path.join(eval_dir, "build", "*", "imce.cpp")):
        reg_func = os.path.basename(os.path.dirname(cpp))
        try:
            body = open(cpp).read()
        except OSError:
            continue
        secs = re.split(r"(?:else )?if \(hid == (\d+) && wid == (\d+)\)", body)
        nodes = {}
        for i in range(1, len(secs) - 2, 3):
            h, w, sec = secs[i], secs[i + 1], secs[i + 2]
            tags = []
            for m in re.finditer(r"// generate: ([a-z_A-Z.]+)", sec):
                g = m.group(1)
                tag = ("qconv" if g.startswith("conv") else
                       "dwconv" if "dwconv" in g else
                       "bn" if g.startswith("batch_norm") else
                       "mm" if g.startswith("min_max") else
                       "add" if g.startswith("add") else
                       "mul" if g.startswith("multl") else None)
                if tag and tag not in tags:
                    tags.append(tag)
            if tags:
                nodes[f"imce_{h}_{w}"] = tags
        out[reg_func] = nodes
    return out

def parse_split_info(eval_dir):
    """{ region_func: { custom_id(int): num_splits } }"""
    txt = _read(eval_dir, "split_info.txt") or "{}"
    out = {}
    try:
        d = ast.literal_eval(txt)
        for reg, convs in d.items():
            out[reg] = {int(cid): info.get("num_splits", 1) for cid, info in convs.items()}
    except Exception:
        pass
    return out

def parse_region_bodies(eval_dir):
    """From 15_with_mappings.txt: per region_func, ordered list of qconv/qdwconv
    dicts (cid, kind, oc, ic, k, src_var, split_fed), the set of custom_ids in the
    body, and the set of vars produced by a split node (to mark split-fed convs)."""
    txt = _read(eval_dir, "15_with_mappings.txt") or ""
    regions = {}; cur = None
    for line in txt.splitlines():
        dm = re.search(r"def @(tvmgen\S*region\d+_main_\d+)\(", line)
        if dm:
            cur = dm.group(1)
            regions.setdefault(cur, {"convs": [], "cids": set(), "split_vars": set()})
            continue
        if cur is None:
            continue
        for cid in re.findall(r"custom_id=(-?\d+)", line):
            regions[cur]["cids"].add(int(cid))
        # record split outputs: "%N = split(%M, ...)" -> the tuple %N (its .k items feed convs)
        sm = re.search(r"%(\d+)\s*=\s*split\(", line)
        if sm:
            regions[cur]["split_vars"].add("%" + sm.group(1))
        # bare add ops with their output channel count (residual adds; used to
        # label a standalone-add imce by its TRUE block instead of nearest-conv)
        am = re.search(r"=\s*add\(%\w+,\s*%\w+.*?custom_id=(\d+)\).*?ty=Tensor\[\(1,\s*(\d+),", line)
        if am:
            regions[cur].setdefault("adds", {})[int(am.group(1))] = int(am.group(2))
        qm = re.search(r"nn\.imcflow_(qconv|qdwconv)\(%(\w+),.*?custom_id=(\d+).*?"
                       r"channels=(\d+),\s*in_channels=(\d+),\s*kernel_size=\[(\d+),\s*(\d+)\]",
                       line)
        if qm:
            src = "%" + qm.group(2)
            # split-fed if the conv's data input is a split-tuple element (%N.k) or the split var
            split_fed = any(src.startswith(sv) for sv in regions[cur]["split_vars"])
            regions[cur]["convs"].append({
                "cid": int(qm.group(3)), "kind": qm.group(1),
                "oc": int(qm.group(4)), "ic": int(qm.group(5)),
                "k": (int(qm.group(6)), int(qm.group(7))),
                "split_fed": split_fed,
            })
    return regions

def parse_noc_paths(eval_dir):
    """{ region_func: [ (src_short, dst_short, dst_tensor_type, hops_or_None, is_dist) ] }"""
    txt = _read(eval_dir, "noc_paths.txt") or ""
    out = {}; cur = None
    for line in txt.splitlines():
        s = line.strip()
        rm = re.match(r"(tvmgen\S*region\d+_main_\d+)$", s)
        if rm:
            cur = rm.group(1); out.setdefault(cur, []); continue
        if cur is None:
            continue
        endpt = re.search(r"\(<NodeID\.(\w+):[^>]*>,\s*<NodeID\.(\w+):[^>]*>,\s*(\w+|None|-?\d+)\)\s*$", s)
        if not endpt:
            continue
        src, dst, hop = endpt.group(1), endpt.group(2), endpt.group(3)
        hops = int(hop) if re.fullmatch(r"-?\d+", hop) else None
        is_dist = s.startswith("NodeID.")
        dtt = None
        if not is_dist:
            # destination tensor-type = the 2nd TensorEdge tuple's role token
            te = re.match(r"TensorEdge\(.*?,\s*\(\(?[-\d, ]+\)?,\s*(\w+)", s)
            dtt = te.group(1) if te else None
        out[cur].append((src, dst, dtt, hops, is_dist))
    return out

# ------------------------- layer naming (graceful) -------------------------

def region_short(func):
    m = re.search(r"region(\d+)", func); return "region" + m.group(1) if m else func

def _family(model):
    if model.startswith("resnet8"): return "resnet8"
    if model.startswith(("ds_cnn", "kws")): return "dscnn"
    if model.startswith(("vww", "mobilenet")): return "mobilenet"
    return "generic"

# ResNet8 layer key: classify EACH conv by its OWN channel signature, NOT by the
# region it landed in. This is residual-in-region-safe: when partitioning merges
# multiple blocks (e.g. b1+b2) into one region, each conv still gets its true
# block label instead of every conv collapsing to the region's block.
#   block  <- out_channels: 16 -> b1, 32 -> b2, 64 -> b3
#   role   <- 1x1 -> .down (shortcut) ; 3x3 full-ic -> .c1 ; 3x3 partial-ic -> .c2
# "full-ic" is the max ic among the 3x3 convs producing the SAME oc (the block's
# first conv uses the full prev-layer channels; atom-split c2 convs have a smaller
# per-atom ic). Ties (block1's two equal 16->16 3x3 with no split) break by cid:
# first = c1, rest = c2.
_RESNET8_BLK_BY_OC = {16: "b1", 32: "b2", 64: "b3"}

def _resnet8_region_layers(reg_short, convs):
    if not convs:
        return None  # no-conv region (e.g. a standalone residual add) -> caller falls back
    out = {}
    # group 3x3 convs by their output-channel block so full-ic detection is per-block
    threes_by_oc = {}
    for c in sorted(convs, key=lambda c: c["cid"]):
        blk = _RESNET8_BLK_BY_OC.get(c["oc"])
        if blk is None:
            continue  # unknown oc -> leave to caller's op-label fallback
        if c["k"] == (1, 1):
            out[c["cid"]] = f"{blk}.down"
        else:
            threes_by_oc.setdefault(c["oc"], []).append(c)
    for oc, threes in threes_by_oc.items():
        blk = _RESNET8_BLK_BY_OC[oc]
        full_ic = max(c["ic"] for c in threes)
        equal_io = all(t["ic"] == t["oc"] for t in threes) and not any(
            t.get("split_fed") for t in threes)
        c1_done = False
        for c in threes:  # already cid-sorted (dict preserves insertion order)
            if equal_io:
                # block1-style: no split, first cid = c1, rest = c2
                out[c["cid"]] = f"{blk}.c1" if not c1_done else f"{blk}.c2"
                c1_done = True
            elif not c1_done and c["ic"] == full_ic and not c.get("split_fed"):
                out[c["cid"]] = f"{blk}.c1"; c1_done = True
            else:
                out[c["cid"]] = f"{blk}.c2"
    return out or None

# ------------------------- extraction -------------------------

# op-type -> short op tag
OP_TAG = {
    "nn.imcflow_qconv": "qconv", "nn.imcflow_qdwconv": "dwconv",
    "imcflow.fused_batch_norm": "bn", "qnn.imcflow_min_max_quantize": "minmax",
    "imcflow.preop-minmax": "minmax", "imcflow.vecops": "vecops",
    "imcflow.qconv2d-with-postop": "qconv", "imcflow.qdwconv2d-with-postop": "dwconv",
    "add": "add", "split": "split",
    "imcflow.qconv2d-split-concat": "split", "imcflow.qdwconv2d-split-concat": "split",
    "imcflow.bn-minmax": "minmax",
}
def _optag(name):
    if name in OP_TAG: return OP_TAG[name]
    for k, v in OP_TAG.items():
        if k in (name or ""): return v
    # fall back to a compact token of the op name
    return (name or "op").split(".")[-1][:8]

def extract(eval_dir, model):
    active = parse_active_imce(eval_dir)
    hwmap = parse_hwnodemap(eval_dir)         # cid -> [nodes]
    id2name = parse_custom_id_to_name(eval_dir)
    composite_patterns = parse_composite_patterns(eval_dir)
    impl_ops = parse_impl_ops(eval_dir)
    splits = parse_split_info(eval_dir)
    bodies = parse_region_bodies(eval_dir)
    fam = _family(model)

    # invert hwmap scoped by region: for a given region, node -> [cids present in that region body]
    result = {"regions": {}}
    all_layers_order = []
    # cross-region conv counters for graceful family naming (dscnn/mobilenet):
    # depthwise+pointwise blocks numbered in global topo (cid) order.
    _blk = {"pw": 0, "dw": 0, "conv": 0}

    for reg_func, imce_set in active.items():
        rshort = region_short(reg_func)
        body = bodies.get(reg_func, {"convs": [], "cids": set()})
        reg_cids = body["cids"]
        rsplit = splits.get(reg_func, {})

        # node -> list of (cid, optype) for cids that belong to THIS region
        node_ops = {n: [] for n in imce_set}
        for cid, nodes in hwmap.items():
            nm = id2name.get(cid, "")
            if cid not in reg_cids:
                # composite CALL cids never print a custom_id= in the relay text,
                # so they miss reg_cids; admit them when their pattern-carrying
                # fn-defn cid (nearest below) IS in this region.
                is_comp = ("with-postop" in nm or "vecops" in nm or "preop" in nm)
                below = [k for k in composite_patterns if k < cid] if is_comp else []
                if not (below and max(below) in reg_cids):
                    continue
            for n in nodes:
                if n in node_ops:
                    node_ops[n].append((cid, nm))

        # conv cid -> human layer name (family heuristic; else op label)
        conv_layer = {}
        conv_order = []
        r8map = _resnet8_region_layers(rshort, body["convs"]) if fam == "resnet8" else None
        for i, conv in enumerate(sorted(body["convs"], key=lambda c: c["cid"])):
            lname = (r8map or {}).get(conv["cid"]) if r8map else None
            if lname is None:
                if conv["kind"] == "qdwconv":
                    base = "dw"
                elif conv["kind"] == "qconv" and conv["k"] == (1, 1):
                    base = "pw"
                else:
                    base = "conv"
                if fam in ("dscnn", "mobilenet"):
                    # global block numbering (dw{n}/pw{n}) across regions -> matches
                    # the DS-CNN dw1/pw1/dw2/pw2... layer naming.
                    _blk[base] += 1
                    lname = f"{base}{_blk[base]}"
                else:
                    lname = f"{rshort}.{base}{i+1}"
            conv_layer[conv["cid"]] = lname
            if lname not in conv_order:
                conv_order.append(lname)

        # region4-style no-conv region -> functional 'residual/add'
        default_layer = conv_order[0] if conv_order else f"{rshort}.op"

        # assign each active imce a layer:
        #  - if it hosts a conv cid -> that conv's layer + split_part
        #  - else attribute to the nearest conv layer by cid proximity in region body
        imces_out = {}
        # order convs by cid to find "nearest conv" for standalone ops
        conv_by_cid = sorted(conv_layer.items())
        for node in sorted(imce_set):
            ops_cids = node_ops[node]
            tags = []
            layer = None; split_part = None
            conv_cids_here = [c for c, nm in ops_cids if _optag(nm) in ("qconv", "dwconv")]
            for cid, nm in sorted(ops_cids):
                # composite CALL cids differ from the pattern-carrying fn-defn
                # cids (the defn prints just before its call), so match a
                # composite call to the nearest pattern cid below it.
                chain = composite_patterns.get(cid)
                if chain is None and ("with-postop" in nm or "vecops" in nm or "preop" in nm):
                    below = [k for k in composite_patterns if k < cid]
                    if below:
                        chain = composite_patterns[max(below)]
                if chain:
                    tags.extend(chain)  # fused composite -> full op chain
                else:
                    tags.append(_optag(nm))
            # dedup tags preserving order
            seen = set(); tags = [t for t in tags if not (t in seen or seen.add(t))]
            # generated-code ground truth overrides (keeps routing-only 'split')
            impl = impl_ops.get(reg_func, {}).get(node)
            if impl:
                tags = impl + (["split"] if "split" in tags else [])
            if conv_cids_here:
                ccid = conv_cids_here[0]
                layer = conv_layer.get(ccid, default_layer)
                # A conv is split if its LAYER appears on >1 active imce that host a
                # qconv (atom split). num_splits comes from split_info keyed on the
                # region's split node cid; if the layer maps to N conv-hosting imces,
                # number them 1..N by imce-id order.
                layer_conv_nodes = sorted(
                    nn for nn, cinfo in [] )  # placeholder, filled after loop
                ns = None
                for scid, nsp in rsplit.items():
                    if nsp and nsp > 1:
                        ns = max(ns or 0, nsp)
                # defer split_part numbering to a post-pass (need all nodes' layers)
            else:
                # standalone RESIDUAL add: label by its TRUE block (output channels
                # 16/32/64 -> b1/b2/b3), not the nearest conv -- e.g. the b2 residual
                # add placed at region3's head must read "b2.res", not "b3.c1".
                body_adds = body.get("adds", {})
                add_blk = None
                if fam == "resnet8":
                    for cid, nm in ops_cids:
                        if _optag(nm) == "add" and cid in body_adds:
                            add_blk = _RESNET8_BLK_BY_OC.get(body_adds[cid])
                            if add_blk:
                                break
                if add_blk:
                    layer = f"{add_blk}.res"
                # otherwise attribute to the nearest conv layer by cid proximity
                elif ops_cids and conv_by_cid:
                    mycid = min(c for c, _ in ops_cids)
                    nearest = min(conv_by_cid, key=lambda kv: abs(kv[0] - mycid))
                    layer = nearest[1]
                else:
                    layer = default_layer
            if not tags:
                tags = ["route"]
            imces_out[node] = {"layer": layer, "ops": tags, "split_part": split_part}

        # split_part post-pass: a layer whose qconv is atom-split lives on >1 imce
        # that host a 'qconv' op. Number those 1..N by imce-id order; N from split_info.
        max_ns = max([n for n in rsplit.values() if n and n > 1] or [0])
        by_layer = {}
        for nn2, cell in imces_out.items():
            if "qconv" in cell["ops"] or "dwconv" in cell["ops"]:
                by_layer.setdefault(cell["layer"], []).append(nn2)
        for lname, nodes in by_layer.items():
            if len(nodes) > 1:
                ns = max_ns if max_ns >= len(nodes) else len(nodes)
                for i, nn2 in enumerate(sorted(nodes), 1):
                    imces_out[nn2]["split_part"] = f"{i}/of-{ns}"

        layers_here = []
        for n in sorted(imces_out):
            l = imces_out[n]["layer"]
            if l not in layers_here:
                layers_here.append(l)
        for l in layers_here:
            if l not in all_layers_order:
                all_layers_order.append(l)

        result["regions"][rshort] = {"imces": imces_out, "layers": layers_here}

    result["layer_order"] = all_layers_order
    counts = {}
    for r in result["regions"].values():
        for c in r["imces"].values():
            counts[c["layer"]] = counts.get(c["layer"], 0) + 1
    result["layer_imce_counts"] = counts
    return result

# ------------------------- XY routing / link channels -------------------------

def _coord(short):
    m = re.match(r"(imce|inode)_(\d+)_(\d+)", short)
    r = int(m.group(2)); c = int(m.group(3)) if m.group(1) == "imce" else 0
    return (r, c)

def _router(short):
    r, c = _coord(short); return f"R{r}_{c}"

def _xy_path(src, dst):
    (r0, c0), (r1, c1) = _coord(src), _coord(dst)
    path = [(r0, c0)]; r, c = r0, c0
    while c != c1:
        c += 1 if c < c1 else -1; path.append((r, c))
    while r != r1:
        r += 1 if r < r1 else -1; path.append((r, c))
    return [f"R{rr}_{cc}" for rr, cc in path]

def _classify(dtt, src_is_inode, is_dist):
    if is_dist:
        return "data"
    t = (dtt or "").lower()
    if t in ("weight", "config") or t in ("min", "max", "fused_scale", "fused_bias", "scale", "bias"):
        return "weight"
    if "func_out" in t:
        return "output"
    # Residual converge: lhs/rhs are the two operands of a residual-add. The
    # SKIP operand arrives from an inode (the residual tensor was buffered in
    # inode memory across the diverge, then re-sent to the add) -> a distinct
    # "residual" channel so the skip-path movement is visible instead of being
    # lumped with ordinary conv input feeds. The MAIN operand arrives from the
    # upstream conv's imce -> psum (same as a normal on-chip producer->consumer).
    if t in ("lhs", "rhs"):
        return "residual" if src_is_inode else "psum"
    if t == "data":
        return "input" if src_is_inode else "psum"
    return "data"

def build_links(eval_dir, mapping):
    paths = parse_noc_paths(eval_dir)
    out = {}
    grid = {"rows": 4, "cols": 5,
            "note": "col0=inode router (inode_r_0), col1-4=imce routers (imce_r_1..4); "
                    "XY dimension-order routing: column(horizontal) first then row(vertical)"}
    for reg_func, edges in paths.items():
        rshort = region_short(reg_func)
        # dedup raw edges by (src,dst,kind)
        agg = {}
        for src, dst, dtt, hops, is_dist in edges:
            kind = _classify(dtt, src.startswith("inode"), is_dist)
            key = (src, dst, kind)
            agg.setdefault(key, {"count": 0, "hops": hops})
            agg[key]["count"] += 1
            if hops is not None:
                agg[key]["hops"] = max(agg[key]["hops"] or 0, hops)
        edges_out = []; links = {}
        for (src, dst, kind), meta in agg.items():
            path = _xy_path(src, dst)
            e = {"src": src, "dst": dst, "kind": kind, "count": meta["count"],
                 "path": path, "hops": len(path) - 1}
            edges_out.append(e)
            for a, b in zip(path, path[1:]):
                lk = f"{a}->{b}"
                links.setdefault(lk, {})
                links[lk][kind] = links[lk].get(kind, 0) + meta["count"]
                links[lk]["total"] = links[lk].get("total", 0) + meta["count"]
        out[rshort] = {"router_grid": grid, "edges": edges_out, "links": links}
    return out

# ------------------------- plotting (adapted from _rtllog/plot_*.py) --------

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
           "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac",
           "#86bcb6", "#d37295"]

def _colors(minfo):
    lo = minfo.get("layer_order") or sorted(
        {l for r in minfo["regions"].values() for l in r["layers"]})
    return {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(lo)}, lo

def plot_region_layer_map(model, minfo, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.patches import Rectangle, FancyArrowPatch
    EDGE_STYLE = {
        "input":  dict(color="#1f77b4", lw=2.2, ls="-",  z=6),
        "weight": dict(color="#7f7f7f", lw=1.2, ls="--", z=4),
        "psum":   dict(color="#d62728", lw=1.6, ls="-",  z=5),
        "output": dict(color="#2ca02c", lw=2.6, ls="-",  z=7),
        "residual": dict(color="#e377c2", lw=2.8, ls="-", z=8)}
    def nc(name):
        p = name.split("_"); row, col = int(p[1]), int(p[2])
        return (-1.0, (3 - row) + 0.46) if name.startswith("inode") else (col - 1 + 0.46, (3 - row) + 0.46)
    color, lo = _colors(minfo)
    regions = minfo["regions"]; n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6))
    if n == 1: axes = [axes]
    fig.suptitle(f"{model} — region mapping (same NN layer = same color)", fontsize=15, fontweight="bold")
    for ax, (rn, ri) in zip(axes, regions.items()):
        imces = ri["imces"]
        ax.set_title(f"{rn}  ({len(ri['layers'])} layers, {len(imces)}/16 IMCE)", fontsize=12)
        for row in range(4):
            y = 3 - row
            ax.add_patch(Rectangle((-1.45, y), 0.9, 0.92, facecolor="#e8e8e8", edgecolor="#999", lw=0.8))
            ax.text(-1.0, y + 0.46, f"inode\n{row}_0", ha="center", va="center", fontsize=7.5, color="#666")
        for row in range(4):
            for col in range(1, 5):
                key = f"imce_{row}_{col}"; x, y = col - 1, 3 - row; cell = imces.get(key)
                if cell:
                    ax.add_patch(Rectangle((x, y), 0.92, 0.92, facecolor=color[cell["layer"]],
                                           edgecolor="black", lw=1.2, alpha=0.9))
                    sp = (cell.get("split_part") or "").replace("/of-", "/")
                    lab = cell["layer"] + (f" {sp}" if sp else "")
                    ax.text(x + 0.46, y + 0.60, lab, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
                    ax.text(x + 0.46, y + 0.30, ",".join(cell["ops"]), ha="center", va="center", fontsize=6.8, color="white")
                else:
                    ax.add_patch(Rectangle((x, y), 0.92, 0.92, facecolor="#f7f7f7", edgecolor="#ccc", lw=0.8))
        for col in range(1, 5):
            ax.text(col - 1 + 0.46, -0.28, f"col{col}", ha="center", fontsize=8, color="#555")
        for row in range(4):
            ax.text(-1.75, 3 - row + 0.46, f"row{row}", va="center", ha="center", fontsize=8, color="#555", rotation=90)
        for i, e in enumerate(ri.get("noc_edges", [])):
            st = EDGE_STYLE.get(e.get("kind"))
            if not st: continue
            (x0, y0), (x1, y1) = nc(e["src"]), nc(e["dst"])
            cnt = e.get("count", 1); rad = 0.18 + 0.06 * (i % 4)
            if (x0, y0) > (x1, y1): rad = -rad
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                         arrowstyle="-|>", mutation_scale=11, shrinkA=11, shrinkB=11, color=st["color"],
                         lw=min(st["lw"] * (1 + 0.25 * (cnt - 1)), 4.5), linestyle=st["ls"], alpha=0.85, zorder=st["z"]))
        ax.set_xlim(-1.9, 4.1); ax.set_ylim(-0.55, 4.05); ax.set_aspect("equal"); ax.axis("off")
    counts = minfo.get("layer_imce_counts", {})
    handles = [Rectangle((0, 0), 1, 1, facecolor=color[l], edgecolor="black") for l in lo]
    labels = [f"{l} ({counts[l]} IMCE)" if l in counts else l for l in lo]
    eh = [mlines.Line2D([], [], color=s["color"], lw=s["lw"], ls=s["ls"]) for s in EDGE_STYLE.values()]
    leg1 = fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.add_artist(leg1)
    fig.legend(eh, [f"NoC: {k}" for k in EDGE_STYLE], loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, -0.025, "('data' program/policy broadcast inode->all-imce omitted for clarity)", ha="center", fontsize=8, color="#777")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    paths = []
    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"region_layer_map_{model}.{ext}"); fig.savefig(p, dpi=150, bbox_inches="tight"); paths.append(p)
    plt.close(fig); return paths

def plot_noc_mesh(model, minfo, links, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.patches import Rectangle, FancyArrow
    KIND = {"data": dict(color="#c9c2b8", base=0.7, scale=0.20),
            "weight": dict(color="#8a8a8a", base=0.9, scale=0.35),
            "input": dict(color="#1f77b4", base=1.2, scale=0.5),
            "psum": dict(color="#d62728", base=1.2, scale=0.5),
            "output": dict(color="#2ca02c", base=1.4, scale=0.5),
            "residual": dict(color="#e377c2", base=1.5, scale=0.6)}
    P, NB, ROFF = 1.5, 1.02, 0.14
    def rxy(r, c): return c * P, (3 - r) * P
    color, lo = _colors(minfo); counts = minfo.get("layer_imce_counts", {})
    regions = minfo["regions"]; n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 6.9))
    if n == 1: axes = [axes]
    fig.suptitle(f"{model} — NoC mesh: region mapping + link-channel occupancy", fontsize=15, fontweight="bold")
    for ax, (rn, ri) in zip(axes, regions.items()):
        imces = ri["imces"]
        ax.set_title(f"{rn}  ({len(ri['layers'])} layers, {len(imces)}/16 IMCE)", fontsize=12)
        for r in range(4):
            for c in range(5):
                x, y = rxy(r, c)
                if c < 4: ax.plot([x, x + P], [y, y], color="#ddd6cc", lw=7, solid_capstyle="round", zorder=1)
                if r < 3: ax.plot([x, x], [y, y - P], color="#ddd6cc", lw=7, solid_capstyle="round", zorder=1)
        # Nodes that participate in a residual-add: the imce that HOSTS the add
        # (residual-edge dst) and the inode that BUFFERS+re-sends the skip
        # (residual-edge src). Marked with a magenta ring + "RES" so the add and
        # its skip-source inode are explicit.
        res_imce, res_inode = set(), set()
        for e in links.get(rn, {}).get("edges", []):
            if e["kind"] == "residual":
                if e["dst"].startswith("imce"): res_imce.add(e["dst"])
                if e["src"].startswith("inode"): res_inode.add(e["src"])
        lk = links.get(rn, {}).get("links", {})
        for key, kinds in lk.items():
            m = re.match(r"R(\d)_(\d)->R(\d)_(\d)", key)
            r0, c0, r1, c1 = map(int, m.groups())
            (x0, y0), (x1, y1) = rxy(r0, c0), rxy(r1, c1)
            horiz = (r0 == r1); fwd = (c1 > c0) if horiz else (r1 > r0); side = 1 if fwd else -1; idx = 0
            for kind, st in KIND.items():
                cnt = kinds.get(kind, 0)
                if not cnt: continue
                off = side * (0.10 + 0.055 * idx); idx += 1; lw = st["base"] + st["scale"] * min(cnt, 8)
                # channels drawn ABOVE node blocks (z=5/6) so arrows are never
                # hidden by the tiles; residual channel sits on top of all others.
                zc = 12 if kind == "residual" else 10
                za = zc + 1
                if horiz:
                    ax.plot([x0 + 0.16, x1 - 0.16], [y0 + off, y1 + off], color=st["color"], lw=lw, alpha=0.9, zorder=zc, solid_capstyle="butt")
                    ax.add_patch(FancyArrow((x0 + x1) / 2, y0 + off, 0.14 * (1 if c1 > c0 else -1), 0, width=0, head_width=0.085, head_length=0.085, color=st["color"], zorder=za, length_includes_head=True))
                else:
                    ax.plot([x0 + off, x1 + off], [y0 - 0.16, y1 + 0.16], color=st["color"], lw=lw, alpha=0.9, zorder=zc, solid_capstyle="butt")
                    ax.add_patch(FancyArrow(x0 + off, (y0 + y1) / 2, 0, 0.14 * (1 if y1 > y0 else -1), width=0, head_width=0.085, head_length=0.085, color=st["color"], zorder=za, length_includes_head=True))
        for r in range(4):
            for c in range(5):
                x, y = rxy(r, c); bx, by = x + ROFF, y + ROFF
                is_res_inode = (c == 0 and f"inode_{r}_0" in res_inode)
                is_res_imce = (c != 0 and f"imce_{r}_{c}" in res_imce)
                # residual participants get a thick magenta ring + a "RES" tag
                ec = "#e377c2" if (is_res_inode or is_res_imce) else None
                elw = 2.6
                if c == 0:
                    ax.add_patch(Rectangle((bx, by), NB, NB, facecolor="#cfe3dd", edgecolor=ec or "#333", lw=elw if ec else 1.2, zorder=5))
                    ax.text(bx + NB / 2, by + NB / 2, f"INODE\n#{r}", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
                    if is_res_inode:
                        ax.text(bx + NB * 0.5, by + NB * 0.15, "RES skip", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#c2185b", zorder=14)
                else:
                    cell = imces.get(f"imce_{r}_{c}")
                    if cell:
                        sp = (cell.get("split_part") or "").replace("/of-", "/"); lab = cell["layer"] + (f" {sp}" if sp else "")
                        lcol = color[cell["layer"]]
                        hosts_conv = any(t in ("qconv", "dwconv") for t in cell["ops"])
                        if hosts_conv:
                            # conv-hosting core: solid layer color, layer name primary
                            ax.add_patch(Rectangle((bx, by), NB, NB, facecolor=lcol, edgecolor=ec or "black", lw=elw if ec else 1.3, alpha=0.92, zorder=5))
                            ax.text(bx + NB / 2, by + NB * 0.62, lab, ha="center", va="center", fontsize=8.2, fontweight="bold", color="white", zorder=6)
                            ax.text(bx + NB / 2, by + NB * 0.32, ",".join(cell["ops"]), ha="center", va="center", fontsize=6.4, color="white", zorder=6)
                        else:
                            # standalone vector-op core (bn/minmax/add/...): hatched tile
                            # tinted in the attributed layer's color, OP NAME primary --
                            # independent non-conv nodes read distinctly from conv hosts.
                            ax.add_patch(Rectangle((bx, by), NB, NB, facecolor="white", edgecolor=ec or lcol, lw=elw if ec else 1.6, hatch="///", zorder=5))
                            ax.add_patch(Rectangle((bx, by), NB, NB, facecolor=lcol, edgecolor="none", alpha=0.18, zorder=5.1))
                            ax.text(bx + NB / 2, by + NB * 0.60, "+".join(cell["ops"]), ha="center", va="center", fontsize=8.0, fontweight="bold", color=lcol, zorder=6)
                            ax.text(bx + NB / 2, by + NB * 0.30, lab, ha="center", va="center", fontsize=6.6, color="#444", zorder=6)
                        if is_res_imce:
                            ax.text(bx + NB * 0.5, by + NB * 0.08, "RES add", ha="center", va="center", fontsize=6.2, fontweight="bold", color="#ffe0f0" if hosts_conv else "#c2185b", zorder=14)
                    else:
                        ax.add_patch(Rectangle((bx, by), NB, NB, facecolor="#f4f4f4", edgecolor="#bbb", lw=0.9, zorder=5))
                        ax.text(bx + NB / 2, by + NB / 2, f"IMCE\n#{(r*4)+(c-1)}", ha="center", va="center", fontsize=7, color="#aaa", zorder=6)
                ax.add_patch(Rectangle((x - 0.11, y - 0.11), 0.22, 0.22, facecolor="#4a4a4a", edgecolor="black", lw=0.8, zorder=7))
        ax.set_xlim(-0.55, 4 * P + NB + 0.35); ax.set_ylim(-0.55, 3 * P + NB + 0.35); ax.set_aspect("equal"); ax.axis("off")
    used = [l for l in lo if counts.get(l)]
    lh = [Rectangle((0, 0), 1, 1, facecolor=color[l], edgecolor="black") for l in used]
    lh.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#666", hatch="///"))
    leg1 = fig.legend(lh, [f"{l} ({counts[l]})" for l in used] + ["standalone vec-op (bn/minmax/add)"],
                      loc="lower center", ncol=min(len(used) + 1, 6), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.045))
    fig.add_artist(leg1)
    eh = [mlines.Line2D([], [], color=s["color"], lw=2.4) for s in KIND.values()]
    fig.legend(eh, [f"ch: {k}" for k in KIND], loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    paths = []
    for ext in ("png", "svg"):
        p = os.path.join(out_dir, f"noc_mesh_{model}.{ext}"); fig.savefig(p, dpi=150, bbox_inches="tight"); paths.append(p)
    plt.close(fig); return paths

# ------------------------- top-level -------------------------

def infer_model(eval_dir):
    """Short, stable model key for filenames. resnet8*/ds_cnn*/vww* -> family name;
    else the eval-dir base (minus _evl.* suffix)."""
    base = os.path.basename(os.path.normpath(eval_dir)).split("_evl")[0]
    if base.startswith("resnet8"): return "resnet8"
    if base.startswith(("ds_cnn", "kws")): return "ds_cnn"
    if base.startswith(("vww", "mobilenet")): return "vww"
    return base

def generate(eval_dir):
    """Extract + render into <eval_dir>/noc_visualizations/. Returns list of files."""
    model = infer_model(eval_dir)
    out_dir = os.path.join(eval_dir, "noc_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    mapping = extract(eval_dir, model)
    links = build_links(eval_dir, mapping)
    # attach noc_edges (endpoint list) into mapping for the region-map plotter
    for rshort, li in links.items():
        if rshort in mapping["regions"]:
            mapping["regions"][rshort]["noc_edges"] = [
                {"src": e["src"], "dst": e["dst"], "kind": e["kind"], "count": e["count"]}
                for e in li["edges"]]
    written = []
    rl = os.path.join(out_dir, "region_layer_map.json")
    with open(rl, "w") as f:
        json.dump({model: mapping}, f, indent=2); written.append(rl)
    nl = os.path.join(out_dir, "noc_link_channels.json")
    with open(nl, "w") as f:
        json.dump({model: links}, f, indent=2); written.append(nl)
    try:
        written += plot_region_layer_map(model, mapping, out_dir)
        written += plot_noc_mesh(model, mapping, links, out_dir)
    except Exception as e:
        sys.stderr.write(f"[noc_mapping_viz] plotting skipped ({type(e).__name__}: {e})\n")
    return written

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: noc_mapping_viz.py <eval_dir>\n"); sys.exit(2)
    for p in generate(sys.argv[1]):
        print(p)
