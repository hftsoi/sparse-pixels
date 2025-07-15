for dir in */; do
    cd "$dir"

    sed -i '/^report_utilization / s/$/ -hierarchical -hierarchical_depth 1/' vivado_synth.tcl
    vitis_hls -f build_prj.tcl "reset=1 synth=1 csim=1 cosim=0 validation=0 export=0 vsynth=1"
    rm -r myhls_prj/solution1/.autopilot myhls_prj/solution1/impl myhls_prj/solution1/syn/verilog myhls_prj/solution1/syn/vhdl myhls_prj/solution1/solution1* myhls_prj/hls.app myhls_prj/solution1/.debug

    cd ..
done
