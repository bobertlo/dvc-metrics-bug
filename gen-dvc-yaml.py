#!/usr/bin/env python3

simple_targets = [
    "test1"
]

targets = {
}

for t in simple_targets:
    targets[t] = {}

print("stages:")

for t in targets:
    this = targets[t]
    arch = this.get("arch", t)
    source = this.get("source", None)
    if not source:
        source = "data/" + t
    print("  " + t + "-training-gen:")
    print("    cmd: python training-gen.py " + t)
    print("    deps:")
    print("     - " + source)
    print("     - training-gen.py")
    #print("     - automator/cnn/architectures/" + arch + ".py")
    print("    params:")
    print("    - " + t + ".arch")
    print("    - " + t + ".data")
    print("    outs:")
    print("    - data/gen/" + t)
    print("  " + t + "-train:")
    print("    cmd: python train-model.py " + t)
    print("    deps:")
    print("    - data/gen/" + t)
    print("    - train-model.py")
    #print("    - automator/cnn/architectures/" + arch + ".py")
    print("    params:")
    print("    - " + t + ".arch")
    print("    - " + t + ".training")
    print("    outs:")
    print("    - cnndata/" + t)
    print("  " + t + "-eval:")
    print(
        "    cmd: python evaluate-model.py "
        + t
        + " results/"
        + t
        + ".csv metrics/"
        + t
        + ".json"
    )
    print("    deps:")
    print("    - evaluate-model.py")
    print("    - cnndata/" + t)
    print("    outs:")
    print("    - results/" + t + ".csv")
    print("    metrics:")
    print("    - metrics/" + t + ".json")
