import sys
from pathlib import Path
import matplotlib.pyplot as plt
import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(PROJECT_ROOT / "data" / "neurogolf2026" / "neurogolf_utils"))
import neurogolf_utils as ng

ng._NEUROGOLF_DIR = str(PROJECT_ROOT / "data" / "neurogolf2026") + "\\"

task_num = 376
arc_gen_limit = 8
run_scoring = False
onnx_path = PROJECT_ROOT / f"task{task_num:03d}.onnx"

examples = ng.load_examples(task_num)

ng.show_examples(examples["train"])
plt.show()

ng.show_examples(examples["test"])
plt.show()

ng.show_examples(examples["arc-gen"][:arc_gen_limit])
plt.show()


def identity_weight(output_channel, input_channel, offset):
    return 1.0 if output_channel == input_channel and offset == (0, 0) else 0.0


if run_scoring:
    if onnx_path.is_file():
        network = onnx.load(onnx_path)
    else:
        print(f"Warning: {onnx_path} not found, using identity demo network.")
        network = ng.single_layer_conv2d_network(identity_weight, kernel_size=1)
    ng.verify_network(network, task_num, examples)
    plt.show()
