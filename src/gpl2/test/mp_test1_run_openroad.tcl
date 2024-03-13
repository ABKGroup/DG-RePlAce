#source "helpers.tcl"
set LIB_DIR "./Nangate45"
#
set tech_lef "$LIB_DIR/Nangate45_tech.lef"
set std_cell_lef "$LIB_DIR/Nangate45.lef"
set fake_macro_lef "$LIB_DIR/fake_macros.lef"
set liberty_file "$LIB_DIR/Nangate45_fast.lib"
set fake_macro_lib "$LIB_DIR/fake_macros.lib"

set def_file mp_test1.def
set verilog_file mp_test1.v

read_lef $tech_lef
read_lef $std_cell_lef
read_lef $fake_macro_lef
read_liberty $liberty_file
read_liberty $fake_macro_lib
 
set top_module "mp_test1"
set nickname "mp_test1"

read_verilog $verilog_file
link_design $top_module
read_def $def_file -floorplan_initialize

global_placement -density 0.7

write_def ${nickname}_cpu.def

exit
