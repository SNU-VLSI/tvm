#!/usr/bin/env python3
"""Two-panel packed-vs-baseline chart from the RTL profiling JSONs.

Panel 1: busy-time 4-phase breakdown (µs) from rtl_pulse_breakdown.py output.
Panel 2: execute-phase per-core-cycle composition (%) from rtl_exec_profile.py.

Usage:
  python tools/plot_busy_breakdown.py pulse.json exec.json out.png \
      [--order name1,name2,...] [--clk-mhz 100]
JSON keys (run names) become bar labels; use --order to control bar order.
"""
import sys, json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHASES = ['execute', 'weight-load', 'imem-program', 'policy-program']
PCOL = {'execute': '#4878a8', 'weight-load': '#6aa84f',
        'imem-program': '#e8a33d', 'policy-program': '#b8b8b8'}
ECLS = ['imcu', 'vpu', 'noc-wait', 'sync-wait', 'ctrl', 'done']
ECOL = {'imcu': '#2f5f8a', 'vpu': '#6aa84f', 'noc-wait': '#d9694f',
        'sync-wait': '#e8c33d', 'ctrl': '#9d7bb0', 'done': '#c9c9c9'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pulse_json'); ap.add_argument('exec_json'); ap.add_argument('out')
    ap.add_argument('--order', default=''); ap.add_argument('--clk-mhz', type=float, default=100.0)
    a = ap.parse_args()
    pulse = json.load(open(a.pulse_json)); prof = json.load(open(a.exec_json))
    names = a.order.split(',') if a.order else list(pulse)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    # panel 1: busy phases in us
    agg = {}
    for n in names:
        t = {}
        for r in pulse[n]:
            for k, v in r['phases'].items():
                t[k] = t.get(k, 0) + v
        agg[n] = t
    bottom = [0.0] * len(names)
    for ph in PHASES:
        vals = [agg[n].get(ph, 0) / a.clk_mhz for n in names]
        ax1.bar(range(len(names)), vals, 0.62, bottom=bottom, label=ph,
                color=PCOL[ph], edgecolor='white', linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, n in enumerate(names):
        tot = sum(agg[n].values()) / a.clk_mhz
        ax1.text(i, tot + max(bottom) * 0.015, f'{tot:.1f} µs', ha='center',
                 fontsize=10, fontweight='bold')
    ax1.set_ylabel(f'accelerator busy time (µs @ {a.clk_mhz:.0f} MHz)')
    ax1.set_title('busy-time phase breakdown (imcflow_state_o=1)', fontsize=10)
    ax1.set_xticks(range(len(names))); ax1.set_xticklabels(names, fontsize=9)
    ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(bottom) * 1.12)

    # panel 2: execute composition, % of active core-cycles
    bottom = [0.0] * len(names)
    for c in ECLS:
        vals = []
        for n in names:
            cc = prof[n]['core_cycles']; tot = sum(cc.values()) or 1
            vals.append(100.0 * cc.get(c, 0) / tot)
        ax2.bar(range(len(names)), vals, 0.62, bottom=bottom, label=c,
                color=ECOL[c], edgecolor='white', linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 6:
                ax2.text(i, bottom[i] + v / 2, f'{v:.0f}%', ha='center',
                         va='center', fontsize=8, color='white')
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax2.set_ylabel('share of active core-cycles (%)')
    ax2.set_title('EXECUTE-phase composition (per-IMCE time attribution)', fontsize=10)
    ax2.set_xticks(range(len(names))); ax2.set_xticklabels(names, fontsize=9)
    ax2.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax2.set_ylim(0, 100); ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f'saved {a.out}')

if __name__ == '__main__':
    main()
