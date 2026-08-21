#!/bin/bash
# Filter script for dataset evaluation output
# Reads stdin, suppresses verbose lines, and renders a progress bar
# Usage: ssh ... | ./scripts/filter_eval_output.sh

BAR_WIDTH=30

awk -v bar_width="$BAR_WIDTH" '
BEGIN {
    in_final = 0
    final_line_count = 0
    last_progress = ""
}

{
    # --- FINAL RESULTS block passthrough ---
    if (in_final) {
        print
        fflush()
        final_line_count++
        # End block on closing === line (after header + content)
        if (final_line_count > 2 && $0 ~ /^=/) {
            in_final = 0
        }
        next
    }

    # Detect FINAL RESULTS start (the === line before it gets eaten, so detect the title)
    if ($0 ~ /FINAL RESULTS/) {
        if (last_progress != "") {
            printf "\n"
            last_progress = ""
        }
        print ""
        print "========================================"
        print $0
        fflush()
        in_final = 1
        final_line_count = 1
        next
    }

    # --- Progress bar ---
    if ($0 ~ /^Progress:/) {
        match($0, /Progress: ([0-9]+)\/([0-9]+).*Failed: ([0-9]+).*Accuracy: ([0-9.]+)%/, m)
        if (RSTART > 0) {
            current = m[1] + 0
            total = m[2] + 0
            failed = m[3] + 0
            accuracy = m[4]

            if (total > 0) {
                filled = int(current * bar_width / total)
            } else {
                filled = 0
            }

            bar = ""
            for (i = 0; i < filled; i++) bar = bar "="
            if (filled < bar_width) {
                bar = bar ">"
                for (i = filled + 1; i < bar_width; i++) bar = bar " "
            }

            printf "\rEvaluating: [%s] %d/%d | Acc: %s%% | Failed: %d", bar, current, total, accuracy, failed
            fflush()
            last_progress = "yes"

            if (current == total) {
                printf "\n"
                last_progress = ""
            }
        }
        next
    }

    # --- Pass through important lines ---
    if ($0 ~ /Starting evaluation of/) { print; fflush(); next }
    if ($0 ~ /^\[CHIP\]/)              { print; fflush(); next }
    if ($0 ~ /^\[SKIP\]/)              { print; fflush(); next }
    if ($0 ~ /FAILED/)                 { print; fflush(); next }
    if ($0 ~ /^Error:/)                { print; fflush(); next }
    if ($0 ~ /failed.*timeout/)        { print; fflush(); next }

    # --- Suppress everything else ---
}
'
