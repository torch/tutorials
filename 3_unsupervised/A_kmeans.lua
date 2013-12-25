
----------------------------------------------------------------------
-- This script implements the k-means algorithm.
-- 
-- The algorithm is implemented such that it produces filters, that
-- can be used in a convolutional network.
--
-- Eugenio Culurciello, Clement Farabet
----------------------------------------------------------------------

require 'image'
require 'unsup'

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.ascii', 'Dataset URL')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 4, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches')
cmd:option('-nkernels', 1024, 'number of kernels to learn')
cmd:option('-niter', 50, 'nb of k-means iterations')
cmd:option('-batchsize', 1000, 'batch size for k-means\' inner loop')
cmd:option('-nsamples', 500000, 'nb of random training samples')
cmd:option('-statinterval', 5000, 'interval for reporting stats/displaying stuff')
cmd:text()
params = cmd:parse(arg or {})

rundir = cmd:string('psd', params, {dir=true})
params.rundir = params.dir .. '/' .. rundir

if paths.dirp(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)
cmd:addTime('psd')
cmd:log(params.rundir .. '/log.txt', params)

torch.manualSeed(params.seed)
torch.setnumthreads(params.threads)

----------------------------------------------------------------------
-- load data
--
dofile '1_data.lua'

filename = paths.basename(params.datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. params.datafile)
end

dataset = getdata(filename, params.inputsize)
dataset:conv()

----------------------------------------------------------------------
-- run k-means
--
print '==> running k-means'

-- create dataset
data = torch.Tensor(params.nsamples,params.inputsize*params.inputsize)
for i = 1,params.nsamples do
   data[i] = dataset[i][1]
end

-- callback: display kernels
function cb (_,kernels,_)
   win = image.display{image=kernels:reshape(params.nkernels,params.inputsize,params.inputsize),
                       padding=2, symmetric=true, zoom=2, win=win,
                       nrow=math.floor(math.sqrt(params.nkernels)),
                       legend='K-Means Centroids'}
end

-- run k-means
kernels = unsup.kmeans(data, params.nkernels, params.niter, params.batchsize, cb, true)

-- save kernels
file = 'kmeans_'..params.nkernels..'.t7'
print('==> saving centroids to disk: ' .. file)
torch.save(file, kernels)
