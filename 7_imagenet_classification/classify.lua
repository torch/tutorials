-- Imagenet classification with Torch7 demo
require 'loadcaffe'
require 'image'

-- Helper functions

-- Loads the mapping from net outputs to human readable labels
function load_synset()
  local list = {}
  for line in io.lines'synset_words.txt' do
    table.insert(list, string.sub(line,11))
  end
  return list
end

-- Converts an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')*255
  -- RGB2BGR
  local im4 = im3:index(1,torch.LongTensor{3,2,1})
  -- subtract imagenet mean
  return im4 - image.scale(img_mean, 224, 224, 'bilinear')
end

-- Setting up networks and downloading stuff if needed
proto_name = 'deploy.prototxt'
model_name = 'nin_imagenet.caffemodel'
img_mean_name = 'ilsvrc_2012_mean.t7'
image_name = 'Goldfish3.jpg'

prototxt_url = 'http://git.io/vIdRW'
model_url = 'https://www.dropbox.com/s/0cidxafrb2wuwxw/'..model_name
img_mean_url = 'https://www.dropbox.com/s/p33rheie3xjx6eu/'..img_mean_name
image_url = 'http://upload.wikimedia.org/wikipedia/commons/e/e9/Goldfish3.jpg'

if not paths.filep(proto_name) then os.execute('wget '..prototxt_url..' -O '..proto_name) end
if not paths.filep(model_name) then os.execute('wget '..model_url)    end
if not paths.filep(img_mean_name) then os.execute('wget '..img_mean_url) end
if not paths.filep(image_name) then os.execute('wget '..image_url)   end


print '==> Loading network'
-- Using network in network http://openreview.net/document/9b05a3bb-3a5e-49cb-91f7-0f482af65aea
net = loadcaffe.load(proto_name, './nin_imagenet.caffemodel')
net.modules[#net.modules] = nil -- remove the top softmax

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

-- Propagate through the network and sort outputs in decreasing order and show 5 best classes
_,classes = net:forward(I):view(-1):sort(true)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end
