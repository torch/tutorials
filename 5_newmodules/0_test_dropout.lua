
require 'nn'
require 'Dropout'

print '==> testing backprop with Jacobian (finite element)'

-- to test the module, we need to freeze the randomness,
-- as the Jacobian tester expects the output of a module
-- to be deterministic...
-- so the code is the same, except that we only generate
-- the random noise once, for the whole test.
function nn.Dropout.updateOutput(self, input)
   self.noise = self.noise or torch.rand(input:size()) -- uniform noise between 0 and 1
   self.noise:add(1 - self.p):floor()  -- a percentage of noise
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   return self.output
end

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.5
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.Dropout(percentage)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
