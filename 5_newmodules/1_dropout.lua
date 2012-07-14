
-- in this file, we test the dropout module we've defined:
require 'nn'
require 'Dropout'
require 'image'

-- define a dropout object:
n = nn.Dropout(0.5)

-- load an image:
i = image.lena()

-- process the image:
result = n:forward(i)

-- display results:
image.display{image=i, legend='original image'}
image.display{image=result, legend='dropout-processed image'}

-- some stats:
mse = i:dist(result)
print('mse between original imgae and dropout-processed image: ' .. mse)
