#!/bin/bash

# Collect data: batch_size|ttft|tpot|tokens_per_s|e2el_s|out_tput|total_tput
data=()
for f in result_*.log; do
    batch_size=$(echo "$f" | sed 's/result_\([0-9]*\)_.*/\1/')
    ttft=$(grep "Median TTFT" "$f" | awk '{print $NF}')
    tpot=$(grep "Median TPOT" "$f" | awk '{print $NF}')
    e2el_ms=$(grep "Median E2EL" "$f" | awk '{print $NF}')
    out_tput=$(grep "Output token throughput" "$f" | awk '{print $NF}')
    total_tput=$(grep "Total Token throughput" "$f" | awk '{print $NF}')

    # tokens/s per user = 1000 / median_tpot
    tokens_per_s=$(awk "BEGIN {printf \"%.2f\", 1000 / $tpot}")
    e2el_s=$(awk "BEGIN {printf \"%.2f\", $e2el_ms / 1000}")

    data+=("$batch_size|$ttft|$tpot|$tokens_per_s|$e2el_s|$out_tput|$total_tput")
done

# Sort by batch_size
IFS=$'\n' sorted=($(printf '%s\n' "${data[@]}" | sort -t'|' -k1 -n)); unset IFS

# ==================== Table format ====================
c1=10; c2=12; c3=12; c4=14; c5=8; c6=14; c7=14
hline="+-$(printf '%-*s' $c1 '' | tr ' ' '-')-+-$(printf '%-*s' $c2 '' | tr ' ' '-')-+-$(printf '%-*s' $c3 '' | tr ' ' '-')-+-$(printf '%-*s' $c4 '' | tr ' ' '-')-+-$(printf '%-*s' $c5 '' | tr ' ' '-')-+-$(printf '%-*s' $c6 '' | tr ' ' '-')-+-$(printf '%-*s' $c7 '' | tr ' ' '-')-+"

echo "$hline"
printf "| %-${c1}s | %-${c2}s | %-${c3}s | %-${c4}s | %-${c5}s | %-${c6}s | %-${c7}s |\n" \
    "batch_size" "TTFT(ms)" "TPOT(ms)" "tok/s/user" "E2EL(s)" "out_tput" "total_tput"
echo "$hline"
for row in "${sorted[@]}"; do
    IFS='|' read -r bs ttft tpot tps e2el otput ttput <<< "$row"
    printf "| %-${c1}s | %-${c2}s | %-${c3}s | %-${c4}s | %-${c5}s | %-${c6}s | %-${c7}s |\n" \
        "$bs" "$ttft" "$tpot" "$tps" "$e2el" "$otput" "$ttput"
done
echo "$hline"

# ==================== CSV format ====================
echo ""
echo "# CSV output:"
echo "batch_size,TTFT(ms),TPOT(ms),tok/s/user,E2EL(s),Output_tput(tok/s),Total_tput(tok/s)"
for row in "${sorted[@]}"; do
    IFS='|' read -r bs ttft tpot tps e2el otput ttput <<< "$row"
    echo "$bs,$ttft,$tpot,$tps,$e2el,$otput,$ttput"
done
