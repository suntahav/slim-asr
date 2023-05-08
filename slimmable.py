import torch.nn as nn

class SwitchableBatchNorm1d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm1d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm1d(i))
        self.bn = nn.ModuleList(bns)
        #self.width_mult = max(FLAGS.width_mult_list)
        #self.ignore_model_profiling = True

    def forward(self, input,idx):
        #idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y
    
class SwitchableGroupnorm(nn.Module):
    def __init__(self, num_groups,num_features_list):
        super(SwitchableGroupnorm, self).__init__()
        self.num_features_list = num_features_list
        self.num_groups = num_groups
        self.num_features = max(num_features_list)
        gns = []
        for i in num_features_list:
            gns.append(nn.GroupNorm(num_groups,i))
        self.gn = nn.ModuleList(gns)
        #self.width_mult = max(FLAGS.width_mult_list)
        #self.ignore_model_profiling = True

    def forward(self, input,idx):
        #idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.gn[idx](input)
        return y


class SwitchableLayerNorm(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableLayerNorm, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        lns = []
        for i in num_features_list:
            lns.append(nn.LayerNorm(i))
        self.ln = nn.ModuleList(lns)
        #self.width_mult = max(FLAGS.width_mult_list)
        #self.ignore_model_profiling = True

    def forward(self, input,idx):
        #idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.ln[idx](input)
        return y

class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list,sliminput=True,slimoutput = True,bias = True,device = None,dtype = None):
        if(sliminput and slimoutput):
            super(SlimmableLinear, self).__init__(max(in_features_list), max(out_features_list),bias = bias,device=device, dtype=dtype)
        elif(sliminput):
            super(SlimmableLinear, self).__init__(max(in_features_list), out_features_list,bias = bias,device=device, dtype=dtype)
        elif(slimoutput):
            super(SlimmableLinear, self).__init__(in_features_list, max(out_features_list),bias = bias,device=device, dtype=dtype)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.sliminput = sliminput
        self.slimoutput = slimoutput
        if(bias ==False):
            self.bias = None
        #self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input,idx):
        if(self.sliminput and self.slimoutput):
            self.in_features = self.in_features_list[idx]
            self.out_features = self.out_features_list[idx]
            weight = self.weight[:self.out_features, :self.in_features]
        elif(self.sliminput):

            self.in_features = self.in_features_list[idx]
            self.out_features = self.out_features_list
            weight = self.weight[:self.out_features, :self.in_features]
        elif(self.slimoutput):
            self.in_features = self.in_features_list
            self.out_features = self.out_features_list[idx]
            weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)
    

class SlimmableConv1d(nn.Conv1d):
    def __init__(self, in_features_list, out_features_list, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,bias = True,depthwise = False):
        super(SlimmableConv1d, self).__init__(
            max(in_features_list), max(out_features_list), kernel_size, stride,
            padding, dilation, groups,bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.depthwise = depthwise
        if(depthwise):
            self.padding = padding
            self.groups =  max(in_features_list)
        #self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input,idx):
        self.in_channels = self.in_features_list[idx]
        self.out_channels = self.out_features_list[idx]
        if(self.depthwise):
            self.groups = self.in_channels
        weight = self.weight[:self.out_channels, :(self.in_channels//self.groups), :]        
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        return nn.functional.conv1d(input, weight, bias, self.stride,
                                    self.padding, self.dilation, self.groups)