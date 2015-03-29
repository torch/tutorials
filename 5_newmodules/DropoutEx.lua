require 'nn'

local DropoutEx, Parent = torch.class('nn.DropoutEx', 'nn.Module')

function DropoutEx:__init(p)
   Parent.__init(self)
   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<DropoutEx> illegal percentage, must be 0 <= p < 1')
   end
   self.noise = torch.Tensor()
end

function DropoutEx:updateOutput(input)
   self.noise:resizeAs(input)
   if self.p > 0 then
      self.noise:bernoulli(1-self.p)
   else
      self.noise:zero()
   end
   self.output:resizeAs(input):copy(input)
   self.output:cmul(self.noise)
   self.output:div(1-self.p)
   return self.output
end

function DropoutEx:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
   self.gradInput:div(1-self.p)
   return self.gradInput
end
