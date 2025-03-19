import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# print("PyTorch Version:", torch.__version__)
# print("CUDA Available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("GPU Device Name:", torch.cuda.get_device_name(0))
#     print("CUDA Version:", torch.version.cuda)
#     print("Total GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

class DoubleConv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(

            # sameConv [height and width are the same]
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):

    def __init__(self, in_channel=3, out_channel=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature
            
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse the list
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] # access the array linearly
            
            # because the size of maxpool is not always divsible by 2
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channel=1, out_channel=1)
    preds = model(x)
    print(preds.shape, x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
