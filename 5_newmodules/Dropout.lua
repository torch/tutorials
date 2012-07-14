require 'nn'

local Dropout, Parent = torch.class('nn.Dropout', 'nn.Module')

function Dropout:__init(percentage)
   Parent.__init(self)
   self.p = percentage or 0.5
   if self.p > 1 or self.p < 0 then
      error('<Dropout> illegal percentage, must be 0 <= p <= 1')
   end
end

function Dropout:updateOutput(input)
   self.noise = torch.rand(input:size()) -- uniform noise between 0 and 1
   self.noise:add(1 - self.p):floor()  -- a percentage of noise
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   return self.gradInput
end
