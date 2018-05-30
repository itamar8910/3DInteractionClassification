import sys

from openpose_util.run_image_get_humans import save_inspections

save_inspections(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
