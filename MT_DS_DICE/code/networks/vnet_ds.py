"""DuCiSC's batch-norm VNet with its four segmentation heads; no feature-alignment branch.

Parameter names (block_*.conv.<i>, out_conv*) and the module construction order match the
original, so a given seed yields the same initial weights.
"""
from torch import nn


def conv_bn_relu(n_in, n_out, kind="same"):
    if kind == "same":
        conv = nn.Conv3d(n_in, n_out, 3, padding=1)
    elif kind == "down":
        conv = nn.Conv3d(n_in, n_out, 2, padding=0, stride=2)
    else:
        conv = nn.ConvTranspose3d(n_in, n_out, 2, padding=0, stride=2)
    return [conv, nn.BatchNorm3d(n_out), nn.ReLU(inplace=True)]


class Block(nn.Module):
    def __init__(self, n_in, n_out, n_stages=1, kind="same"):
        super().__init__()
        self.conv = nn.Sequential(*[layer for stage in range(n_stages)
                                    for layer in conv_bn_relu(n_in if stage == 0 else n_out, n_out, kind)])

    def forward(self, x):
        return self.conv(x)


class VNetDS(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, has_dropout=False):
        super().__init__()
        self.has_dropout = has_dropout
        f = n_filters
        self.block_one = Block(n_channels, f, 1)
        self.block_one_dw = Block(f, 2 * f, kind="down")
        self.block_two = Block(2 * f, 2 * f, 2)
        self.block_two_dw = Block(2 * f, 4 * f, kind="down")
        self.block_three = Block(4 * f, 4 * f, 3)
        self.block_three_dw = Block(4 * f, 8 * f, kind="down")
        self.block_four = Block(8 * f, 8 * f, 3)
        self.block_four_dw = Block(8 * f, 16 * f, kind="down")
        self.block_five = Block(16 * f, 16 * f, 3)
        self.block_five_up = Block(16 * f, 8 * f, kind="up")
        self.block_six = Block(8 * f, 8 * f, 3)
        self.block_six_up = Block(8 * f, 4 * f, kind="up")
        self.block_seven = Block(4 * f, 4 * f, 3)
        self.block_seven_up = Block(4 * f, 2 * f, kind="up")
        self.block_eight = Block(2 * f, 2 * f, 2)
        self.block_eight_up = Block(2 * f, f, kind="up")
        self.block_nine = Block(f, f, 1)
        self.out_conv = nn.Conv3d(f, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # Deep-supervision heads at 1/2, 1/4 and 1/8 resolution.
        self.out_conv_2 = nn.Conv3d(2 * f, n_classes, 1, padding=0)
        self.out_conv_3 = nn.Conv3d(4 * f, n_classes, 1, padding=0)
        self.out_conv_4 = nn.Conv3d(8 * f, n_classes, 1, padding=0)

    def forward(self, x, deepsup=False, features=False):
        """With features=True also returns dict(x1=x1, x7_up=x7_up) (the same tensors, nothing recomputed)."""
        x1 = self.block_one(x)
        x2 = self.block_two(self.block_one_dw(x1))
        x3 = self.block_three(self.block_two_dw(x2))
        x4 = self.block_four(self.block_three_dw(x3))
        x5 = self.block_five(self.block_four_dw(x4))
        if self.has_dropout:
            x5 = self.dropout(x5)
        out = []
        x5_up = self.block_five_up(x5) + x4
        out.append(self.out_conv_4(x5_up))
        x6_up = self.block_six_up(self.block_six(x5_up)) + x3
        out.append(self.out_conv_3(x6_up))
        x7_up = self.block_seven_up(self.block_seven(x6_up)) + x2
        out.append(self.out_conv_2(x7_up))
        x8_up = self.block_eight_up(self.block_eight(x7_up)) + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out.append(self.out_conv(x9))
        result = out if deepsup else out[-1]
        return (result, dict(x1=x1, x7_up=x7_up)) if features else result

    def decode_from_x7(self, x7_up, x1, dropout=False):
        """The full-resolution head from D2 (x7_up) on: block_eight → block_eight_up → +x1 → block_nine → out_conv."""
        x9 = self.block_nine(self.block_eight_up(self.block_eight(x7_up)) + x1)
        if dropout and self.has_dropout:
            x9 = self.dropout(x9)
        return self.out_conv(x9)
