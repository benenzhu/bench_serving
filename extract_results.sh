#!/bin/bash

# Collect data: batch_size|ttft|tpot|interactivity|e2el_s|tput_gpu|out_tput_gpu|in_tput_gpu
data=()
for f in result_*.log; do
    batch_size=$(echo "$f" | sed 's/result_\([0-9]*\)_.*/\1/')
    ttft=$(grep "Median TTFT" "$f" | awk '{print $NF}')
    tpot=$(grep "Median TPOT" "$f" | awk '{print $NF}')
    interactivity=$(grep "Tokens/sec TPOT" "$f" | awk '{print $NF}')
    e2el_ms=$(grep "Median E2EL" "$f" | awk '{print $NF}')
    tput_gpu=$(grep "single gpu throughput" "$f" | awk '{print $NF}')
    total_tput=$(grep "Total Token throughput" "$f" | awk '{print $NF}')
    out_tput=$(grep "Output token throughput" "$f" | awk '{print $NF}')

    # Compute derived metrics
    e2el_s=$(awk "BEGIN {printf \"%.2f\", $e2el_ms / 1000}")
    num_gpus=$(awk "BEGIN {printf \"%.0f\", $total_tput / $tput_gpu}")
    out_tput_gpu=$(awk "BEGIN {printf \"%.2f\", $out_tput / $num_gpus}")
    in_tput_gpu=$(awk "BEGIN {printf \"%.2f\", ($total_tput - $out_tput) / $num_gpus}")

    data+=("$batch_size|$ttft|$tpot|$interactivity|$e2el_s|$tput_gpu|$out_tput_gpu|$in_tput_gpu")
done

# Sort by batch_size
IFS=$'\n' sorted=($(printf '%s\n' "${data[@]}" | sort -t'|' -k1 -n)); unset IFS

# ==================== Table format ====================
c1=12; c2=30; c3=17
hline="+-$(printf '%-*s' $c1 '' | tr ' ' '-')-+-$(printf '%-*s' $c2 '' | tr ' ' '-')-+-$(printf '%-*s' $c3 '' | tr ' ' '-')-+"

echo "$hline"
printf "| %-${c1}s | %-${c2}s | %-${c3}s |\n" "batch_size" "single_gpu_throughput(tok/s)" "Tokens/sec_TPOT"
echo "$hline"
for row in "${sorted[@]}"; do
    IFS='|' read -r bs ttft tpot inter e2el tgpu ogpu igpu <<< "$row"
    printf "| %-${c1}s | %-${c2}s | %-${c3}s |\n" "$bs" "$tgpu" "$inter"
done
echo "$hline"

# ==================== CSV format ====================
echo ""
echo "# CSV output:"
echo "batch_size,TTFT(ms),TPOT(ms),Interactivity(tok/s/user),E2EL(s),TPUT_per_GPU(tok/s),Output_TPUT_per_GPU(tok/s),Input_TPUT_per_GPU(tok/s)"
for row in "${sorted[@]}"; do
    IFS='|' read -r bs ttft tpot inter e2el tgpu ogpu igpu <<< "$row"
    echo "$bs,$ttft,$tpot,$inter,$e2el,$tgpu,$ogpu,$igpu"
done
