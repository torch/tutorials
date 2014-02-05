----------------------------------------------------------------------
-- getstarted.lua
-- 
-- This script demonstrates very basic Lua/Torch stuff.

-- To run this script, simply do:
-- torch getstarted.lua
-- and then press 'y' or 'return' at each step, to keep going, or 'n'
-- to terminate.

require 'gfx.go'

----------------------------------------------------------------------
-- little function to pause execution, and request user input
function next()
   local answer = nil
   while answer ~= '' and answer ~= 'y' and answer ~= 'Y' and neverstall ~= true do
      io.write("continue ([y]/n/!)? ")
      io.flush()
      answer=io.read()
      if answer == '!' then
         neverstall = true
      end
      if answer == 'n' then
         print('exiting...')
         os.exit()
      end
   end
   print ''
end

----------------------------------------------------------------------
-- snippet 1
print('basic printing')
a = 10
print(a)
print 'something'
print(type(a))
print(type('something'))
next()

----------------------------------------------------------------------
-- snippet 3
require 'image'
i = image.lena()
gfx.image(i)
next()

----------------------------------------------------------------------
-- snippet 4
require 'nn'
n = nn.SpatialConvolution(1,16,12,12)
gfx.image(n.weight, {zoom=2, legend=''})
next()

----------------------------------------------------------------------
-- snippet 5
res = n:forward(image.rgb2y(i))
gfx.image(res, {zoom=0.25, legend='states'})
next()
