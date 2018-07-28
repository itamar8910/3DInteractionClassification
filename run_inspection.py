import sys

from openpose_util.run_image_get_humans import save_inspections

"usage: cam inndex(0-3), start_sec, end_sec"

save_inspections(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
