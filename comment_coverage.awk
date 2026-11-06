{ sub("%", "", $2); err = $2+0 < thres+0 } END {exit err}
