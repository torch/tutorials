
-- in this file, we test the dropout module we've defined:
require 'nn'
require 'DropoutEx'
require 'image'
require 'gfx.js'

-- define a dropout object:
n = nn.DropoutEx(0.5)

-- load an image:
i = image.lena()

-- process the image:
result = n:forward(i)

-- display results:
gfx.image(i, {legend='original image'})
gfx.image(result, {legend='dropout-processed image'})

-- some stats:
mse = i:dist(result)
print('mse between original imgae and dropout-processed image: ' .. mse)
