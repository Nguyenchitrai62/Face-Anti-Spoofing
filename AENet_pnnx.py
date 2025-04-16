import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(7,7), out_channels=64, padding=(3,3), padding_mode='zeros', stride=(2,2))
        self.conv2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_5 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.conv2d_6 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_7 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(2,2))
        self.conv2d_8 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_9 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_10 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.conv2d_11 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_12 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=256, padding=(0,0), padding_mode='zeros', stride=(2,2))
        self.conv2d_13 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_14 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_15 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.conv2d_16 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_17 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(1,1), out_channels=512, padding=(0,0), padding_mode='zeros', stride=(2,2))
        self.conv2d_18 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.conv2d_19 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.linear_0 = nn.Linear(bias=True, in_features=512, out_features=2)

        archive = zipfile.ZipFile('AENet.pnnx.bin', 'r')
        self.conv2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.bias', (64), 'float32')
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.weight', (64,3,7,7), 'float32')
        self.conv2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_1.bias', (64), 'float32')
        self.conv2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_1.weight', (64,64,3,3), 'float32')
        self.conv2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_2.bias', (64), 'float32')
        self.conv2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_2.weight', (64,64,3,3), 'float32')
        self.conv2d_3.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_3.bias', (64), 'float32')
        self.conv2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_3.weight', (64,64,3,3), 'float32')
        self.conv2d_4.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_4.bias', (64), 'float32')
        self.conv2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_4.weight', (64,64,3,3), 'float32')
        self.conv2d_5.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_5.bias', (128), 'float32')
        self.conv2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_5.weight', (128,64,3,3), 'float32')
        self.conv2d_6.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_6.bias', (128), 'float32')
        self.conv2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_6.weight', (128,128,3,3), 'float32')
        self.conv2d_7.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_7.bias', (128), 'float32')
        self.conv2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_7.weight', (128,64,1,1), 'float32')
        self.conv2d_8.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_8.bias', (128), 'float32')
        self.conv2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_8.weight', (128,128,3,3), 'float32')
        self.conv2d_9.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_9.bias', (128), 'float32')
        self.conv2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_9.weight', (128,128,3,3), 'float32')
        self.conv2d_10.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_10.bias', (256), 'float32')
        self.conv2d_10.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_10.weight', (256,128,3,3), 'float32')
        self.conv2d_11.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_11.bias', (256), 'float32')
        self.conv2d_11.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_11.weight', (256,256,3,3), 'float32')
        self.conv2d_12.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_12.bias', (256), 'float32')
        self.conv2d_12.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_12.weight', (256,128,1,1), 'float32')
        self.conv2d_13.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_13.bias', (256), 'float32')
        self.conv2d_13.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_13.weight', (256,256,3,3), 'float32')
        self.conv2d_14.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_14.bias', (256), 'float32')
        self.conv2d_14.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_14.weight', (256,256,3,3), 'float32')
        self.conv2d_15.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_15.bias', (512), 'float32')
        self.conv2d_15.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_15.weight', (512,256,3,3), 'float32')
        self.conv2d_16.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_16.bias', (512), 'float32')
        self.conv2d_16.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_16.weight', (512,512,3,3), 'float32')
        self.conv2d_17.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_17.bias', (512), 'float32')
        self.conv2d_17.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_17.weight', (512,256,1,1), 'float32')
        self.conv2d_18.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_18.bias', (512), 'float32')
        self.conv2d_18.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_18.weight', (512,512,3,3), 'float32')
        self.conv2d_19.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_19.bias', (512), 'float32')
        self.conv2d_19.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_19.weight', (512,512,3,3), 'float32')
        self.linear_0.bias = self.load_pnnx_bin_as_parameter(archive, 'linear_0.bias', (2), 'float32')
        self.linear_0.weight = self.load_pnnx_bin_as_parameter(archive, 'linear_0.weight', (2,512), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv2d_0(v_0)
        v_2 = F.relu(input=v_1)
        v_3 = F.max_pool2d(input=v_2, ceil_mode=False, dilation=(1,1), kernel_size=(3,3), padding=(1,1), return_indices=False, stride=(2,2))
        v_4 = self.conv2d_1(v_3)
        v_5 = F.relu(input=v_4)
        v_6 = self.conv2d_2(v_5)
        v_7 = (v_6 + v_3)
        v_8 = F.relu(input=v_7)
        v_9 = self.conv2d_3(v_8)
        v_10 = F.relu(input=v_9)
        v_11 = self.conv2d_4(v_10)
        v_12 = (v_11 + v_8)
        v_13 = F.relu(input=v_12)
        v_14 = self.conv2d_5(v_13)
        v_15 = F.relu(input=v_14)
        v_16 = self.conv2d_6(v_15)
        v_17 = self.conv2d_7(v_13)
        v_18 = (v_16 + v_17)
        v_19 = F.relu(input=v_18)
        v_20 = self.conv2d_8(v_19)
        v_21 = F.relu(input=v_20)
        v_22 = self.conv2d_9(v_21)
        v_23 = (v_22 + v_19)
        v_24 = F.relu(input=v_23)
        v_25 = self.conv2d_10(v_24)
        v_26 = F.relu(input=v_25)
        v_27 = self.conv2d_11(v_26)
        v_28 = self.conv2d_12(v_24)
        v_29 = (v_27 + v_28)
        v_30 = F.relu(input=v_29)
        v_31 = self.conv2d_13(v_30)
        v_32 = F.relu(input=v_31)
        v_33 = self.conv2d_14(v_32)
        v_34 = (v_33 + v_30)
        v_35 = F.relu(input=v_34)
        v_36 = self.conv2d_15(v_35)
        v_37 = F.relu(input=v_36)
        v_38 = self.conv2d_16(v_37)
        v_39 = self.conv2d_17(v_35)
        v_40 = (v_38 + v_39)
        v_41 = F.relu(input=v_40)
        v_42 = self.conv2d_18(v_41)
        v_43 = F.relu(input=v_42)
        v_44 = self.conv2d_19(v_43)
        v_45 = (v_44 + v_41)
        v_46 = F.relu(input=v_45)
        v_47 = F.avg_pool2d(input=v_46, ceil_mode=False, count_include_pad=True, divisor_override=None, kernel_size=(7,7), padding=(0,0), stride=(1,1))
        v_48 = v_47.reshape(1, 512)
        v_49 = self.linear_0(v_48)
        return v_49

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 224, 224, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("AENet_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 224, 224, dtype=torch.float)

    torch.onnx.export(net, v_0, "AENet_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 224, 224, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
