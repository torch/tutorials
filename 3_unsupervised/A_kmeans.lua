
----------------------------------------------------------------------
-- This script implements the k-means algorithm.
-- 
-- The algorithm is implemented such that it produces filters, that
-- can be used in a convolutional network.
--
-- Eugenio Culurciello, Clement Farabet
----------------------------------------------------------------------

require 'image'

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-datafile', 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.bin', 'Dataset URL')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 2, 'threads')
cmd:option('-inputsize', 9, 'size of each input patches')
cmd:option('-nkernels', 128, 'number of kernels to learn')
cmd:option('-maxiter', 100000, 'max number of updates')
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
   os.execute('wget ' .. params.datafile .. '; '.. 'tar xvf ' .. filename)
end
dataset = getdata(filename, params.inputsize)
dataset:conv()

----------------------------------------------------------------------
-- run k-means
--
print '==> running k-means'

-- initialize K-means with random data samples:
kernels = torch.Tensor(params.nkernels, params.inputsize, params.inputsize)
kernels_avg = torch.Tensor(params.nkernels)
for t = 1, params.nkernels do
   -- initialize k-means kernels with random data samples:
   while true do
      kernels[t] = dataset[t][1]
      if kernels[t]:std() > 0.2 then -- ignore templates with not enough structure
         break
      end
   end
   kernels_avg[t] = 1
end

-- display initial kernels:
image.display{image=kernels, padding=2, symmetric=true, zoom=2,
              nrow=math.floor(math.sqrt(params.nkernels)),
              legend='Initial Kernels'}

-- find K-means (iteratively)
dists = torch.Tensor(params.nkernels)
for t = 1,params.maxiter do
   -- progress
   xlua.progress(t, params.maxiter)

   -- get next patch
   local sample = dataset[t][1]

   -- try to match new sample with kernels
   for i = 1,params.nkernels do
      dists[i] = torch.dist(sample, kernels[i], 1)
   end

   -- closest template:
   min,argmin = dists:min(1)
   min=min[1]; argmin=argmin[1]

   -- average template:
   kernels[argmin] = (kernels[argmin]*kernels_avg[argmin] + sample)/(kernels_avg[argmin]+1)

   -- update counter:
   kernels_avg[argmin] = kernels_avg[argmin] + 1

   -- stats
   if (t % params.statinterval) == 0 then
      -- normalize kernels:
      kernels_avg_max = kernels_avg:max()
      kernels_normed = kernels:clone()
      for t = 1, params.nkernels do
         kernels_normed[t] = kernels_normed[t]*(kernels_avg[t]/kernels_avg_max)
      end

      -- display new filters:
      win = image.display{image=kernels_normed, padding=2, symmetric=true, zoom=2, win=win,
                          nrow=math.floor(math.sqrt(params.nkernels)),
                          legend='Kernels @ t='..t}

      -- cleanup
      collectgarbage()
   end
end

-- normalize kernels:
kernels_avg_max = kernels_avg:max()
kernels_normed = kernels:clone()
for t = 1, params.nkernels do
   kernels_normed[t] = kernels_normed[t]*(kernels_avg[t]/kernels_avg_max)
end

-- final report
print('==> computed ' .. params.nkernels .. ' kernels')
print('==> least significant kernel has ' .. (kernels_avg:min()-1) .. ' averages')
print('==> most significant kernel has ' .. (kernels_avg:max()-1) .. ' averages')

-- discard filters that are not representative enough:
final_kernels = {}
for i = 1,params.nkernels do
   if kernels_avg[i] > 0.1*kernels_avg_max then
      table.insert(final_kernels, kernels_normed[i])
   end
end
print('==> retaining ' .. #final_kernels .. ' top kernels')

-- display
image.display{image=final_kernels, padding=2, symmetric=true, zoom=2,
              nrow=math.floor(math.sqrt(#final_kernels)),
              legend='Final Kernels'}
