from torchvision import models
import time
from models.layers import *


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.mlfa = Multi_Level_Feature_Aggreagation()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)  # size:1/2
        x = self.maxpool(x)  # size:1/4
        x_low = self.layer1(x)  # size:1/4
        x1 = self.layer2(x_low)  # size:1/8
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x = self.mlfa(x1, x2, x3)
        return x, x_low


class BTSCD(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(BTSCD, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)

        self.change_specific_transfer = Change_Specific_Transfer(128) # BCFE

        self.DecCD = decoder(128, 64)
        self.Dec1 = decoder(128, 64)
        self.Dec2 = decoder(128, 64)

        self.task_interaction = task_interaction_module()

        self.classifierSem1 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem2 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierCD = nn.Conv2d(64, 2, 1, 1, 0, bias=False)

        self.boundary_decoder = Boundary_Decoder()
        self.eca = ECA()
        self.boundary_classifier = nn.Sequential(
            CBA3x3(64, 32),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x_size = x1.size()

        x1, x1_low = self.FCN(x1)
        x2, x2_low = self.FCN(x2)

        xc = self.change_specific_transfer(x1, x2)

        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)

        xc_low = torch.abs(x1 - x2)
        xc = self.DecCD(xc, xc_low)

        #Classifier
        change = self.classifierCD(xc)
        new_xc, pixel_sim_loss = self.task_interaction(x1, x2, xc, change)

        out1 = self.classifierSem1(x1)
        out2 = self.classifierSem2(x2)
        new_change = self.classifierCD(new_xc)

        out1 = F.upsample(out1, x_size[2:], mode='bilinear')
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        change_out = F.upsample(new_change, x_size[2:], mode='bilinear')

        boundary_x1 = self.boundary_decoder(x1, x_size[2:])
        boundary_x2 = self.boundary_decoder(x2, x_size[2:])
        boundary_change = self.boundary_decoder(new_xc, x_size[2:])

        boundary_sem = self.eca(boundary_x1 + boundary_x2)
        boundary_sem = self.boundary_classifier(boundary_sem)
        boundary_change = self.boundary_classifier(boundary_change)

        return change_out, out1, out2, pixel_sim_loss, boundary_sem, boundary_change


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 512, 512).cuda().float()
    x2 = torch.randn(1, 3, 512, 512).cuda().float()

    model = BTSCD(3, num_classes=7).cuda()
    model.eval()  # 将模型设置为推理模式
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, (x1, x2))
    total = sum([param.nelement() for param in model.parameters()])
    print("Params_Num: %.2fM" % (total/1e6))
    print("FLOPs: %.2fG" % (flops.total()/1e9))

    with torch.no_grad():
        for _ in range(10):
            _ = model(x1, x2)

    # 正式计时
    start_time = time.time()
    with torch.no_grad():
        output = model(x1, x2)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000:.2f} ms")
