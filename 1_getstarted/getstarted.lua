----------------------------------------------------------------------
-- getstarted.lua
-- 
-- This script demonstrates very basic Lua/Torch stuff.

-- To run this script, simply do:
-- th getstarted.lua
-- To run the script from itorch, in a notebook do:
-- dofile 'getstarted.lua'

----------------------------------------------------------------------
-- snippet 1
print('basic printing')
a = 10
print(a)
print 'something'
print(type(a))
print(type('something'))

----------------------------------------------------------------------
-- snippet 2
require 'image'
i = image.lena()
if itorch then
   itorch.image(i)
else
   print('skipping visualization because the script is not run from itorch')
end

----------------------------------------------------------------------
-- snippet 4
require 'nn'
n = nn.SpatialConvolution(1,16,12,12)
if itorch then
   itorch.image(n.weight)
else
   print('skipping visualization because the script is not run from itorch')
end

----------------------------------------------------------------------
-- snippet 5
res = n:forward(image.rgb2y(i))
if itorch then
   itorch.image(res)
else
   print('skipping visualization because the script is not run from itorch')
end
