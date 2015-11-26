
-- in this file, we test the dropout module we've defined:
require 'nn'
require 'DropoutEx'
require 'image'

-- define a dropout object:
n = nn.DropoutEx(0.5)

-- load an image:
i = image.lena()

-- process the image:
result = n:forward(i)

if itorch then
   -- display results:
   print('original image:')
   itorch.image(i)
   print('result image:')
   itorch.image(result)
end

-- some stats:
mse = i:dist(result)
print('mse between original imgae and dropout-processed image: ' .. mse)
