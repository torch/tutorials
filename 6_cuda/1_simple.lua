
require 'cutorch'

t1 = torch.randn(1000):cuda()
t2 = torch.randn(1000):cuda()

t1:add(t2)

t1_f = t1:float()

print{t1}
print{t1_f}
