import subprocess

onnx_model = "AENet.onnx"

# Thêm inputshape theo đúng yêu cầu của mô hình
cmd = [
    "./pnnx/pnnx", 
    onnx_model, 
    "inputshape=[1,3,224,224]"
]

subprocess.run(cmd)
