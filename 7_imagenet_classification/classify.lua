-- Imagenet classification with Torch7 demo
require 'loadcaffe'
require 'image'

-- Helper functions

-- Loads the mapping from net outputs to human readable labels
function load_synset()
  local file = io.open 'synset_words.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line,11))
  end
  return list
end


-- Converts an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')*255
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  -- subtract imagenet mean
  return im4 - image.scale(img_mean, 224, 224, 'bilinear')
end



-- Setting up networks and downloading stuff if needed
proto_name = 'deploy.prototxt'
model_name = 'nin_imagenet.caffemodel'
img_mean_name = 'ilsvrc_2012_mean.t7'
image_name = 'Goldfish3.jpg'

prototxt_url = 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/'..proto_name
model_url = 'https://www.dropbox.com/s/0cidxafrb2wuwxw/'..model_name
img_mean_url = 'https://www.dropbox.com/s/p33rheie3xjx6eu/'..img_mean_name
image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'

if not paths.filep(proto_name) then os.execute('wget '..prototxt_url) end
if not paths.filep(model_name) then os.execute('wget '..model_url)    end
if not paths.filep(img_mean_name) then os.execute('wget '..img_mean_url) end
if not paths.filep(image_name) then os.execute('wget '..image_url)   end



print '==> Loading network'
-- we'll use the fastest CUDA ConvNet implementation available, cuda-convnet2
-- this loads the network in Caffe format and returns in Torch format, ready to use!
--net = loadcaffe.load(proto_name, model_name, 'cudnn')
net = loadcaffe.load(proto_name, './nin_imagenet.caffemodel', 'nn')
net.modules[#net.modules] = nn.View(1000):setNumInputDims(3)

-- as we want to classify, let's disable dropouts by enabling evaluation mode
net:evaluate()

print '==> Loading synsets'
synset_words = load_synset()

print '==> Loading image and imagenet mean'
im = image.load(image_name)
img_mean = torch.load(img_mean_name).img_mean:transpose(3,1)

print '==> Preprocessing'
-- Have to resize and convert from RGB to BGR and subtract mean
I = preprocess(im, img_mean)

-- cuda-convnet2 implementation support only batched routines, so
-- we have to allocate memory for 32 inputs and then put crops to 10 of them.
-- let's however use just one image for simplicity.
-- note that for other networks that use cunn ore cudnn that might not be needed
batch = torch.CudaTensor(1,3,224,224)
batch[1]:copy(I)

print '==> Propagating through the network'
net:forward(batch)

-- sort outputs in decreasing order
_,classes = net.output[{1,{}}]:float():sort(true)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end
