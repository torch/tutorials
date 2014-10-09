
require 'nn'
require 'DropoutEx'

print '==> testing backprop with Jacobian (finite element)'

-- to test the module, we need to freeze the randomness,
-- as the Jacobian tester expects the output of a module
-- to be deterministic...
-- so the code is the same, except that we only generate
-- the random noise once, for the whole test.
firsttime = true
function nn.DropoutEx.updateOutput(self, input)
   if firsttime then
      self.noise:resizeAs(input)
      if self.p > 0 then
         self.noise:bernoulli(1-self.p)
      else
         self.noise:zero()
      end
      firsttime = false
   end
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   self.output:div(1-self.p)
   return self.output
end

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local ini = math.random(10,20)
local inj = math.random(10,20)
local ink = math.random(10,20)
local percentage = 0.25
local input = torch.Tensor(ini,inj,ink):zero()
local module = nn.DropoutEx(percentage)

-- test backprop, with Jacobian
local err = jac.testJacobian(module,input)
print('==> error: ' .. err)
if err<precision then
   print('==> module OK')
else
   print('==> error too large, incorrect implementation')
end
