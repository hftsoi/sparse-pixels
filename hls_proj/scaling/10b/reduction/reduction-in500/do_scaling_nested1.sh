for dir in */; do
    cd "$dir"

    sed -i '/^report_utilization / s/$/ -hierarchical -hierarchical_depth 1/' vivado_synth.tcl
    vitis_hls -f build_prj.tcl "reset=1 synth=1 csim=0 cosim=0 validation=0 export=0 vsynth=1"
    rm -r hls_dummy_prj/solution1/.autopilot hls_dummy_prj/solution1/impl hls_dummy_prj/solution1/syn/verilog hls_dummy_prj/solution1/syn/vhdl hls_dummy_prj/solution1/solution1* hls_dummy_prj/hls.app hls_dummy_prj/solution1/.debug

    cd ..
done