----------------------------------------------------------------------
-- This script shows how to train autoencoders on natural images,
-- using the unsup package.
--
-- A lote of this code was written and contributed by Koray Kavukcuoglu.
--
-- In this script, we demonstrate the use of different types of
-- autoencoders. Learned filters can be visualized by providing the
-- flag -display.
--
-- Note: simple auto-encoders (with no sparsity constraint on the code) typically
-- don't yield filters that are visually appealing, although they might be
-- minimizing the reconstruction error correctly.
--
-- We demonstrate 2 types of auto-encoders:
--   * plain: regular auto-encoder
--   * predictive sparse decomposition (PSD): the encoder is trained
--     to predict an optimal sparse decomposition of the input
--
-- Both types of auto-encoders can use linear or convolutional
-- encoders/decoders. The convolutional version typically yields more
-- interesting, less redundant filters for images.
--
-- Koray Kavukcuoglu, Clement Farabet
----------------------------------------------------------------------

require 'unsup'
require 'image'
require 'optim'

----------------------------------------------------------------------
-- parse command-line options
--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple sparse coding dictionary on Berkeley images')
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-dir', 'outputs', 'subdirectory to save experiments in')
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 2, 'threads')

-- for all models:
cmd:option('-model', 'conv-psd', 'auto-encoder class: linear | linear-psd | conv | conv-psd')
cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-nfiltersin', 1, 'number of input convolutional filters')
cmd:option('-nfiltersout', 16, 'number of output convolutional filters')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 1, 'batch size')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-momentum', 0, 'gradient momentum')
cmd:option('-maxiter', 1000000, 'max number of updates')

-- for linear model only:
cmd:option('-tied', false, 'decoder weights are tied to encoder\'s weights (transposed)')

-- use hessian information for training:
cmd:option('-hessian', false, 'compute diagonal hessian coefficients to condition learning rates')
cmd:option('-hessiansamples', 500, 'number of samples to use to estimate hessian')
cmd:option('-hessianinterval', 10000, 'compute diagonal hessian coefs at every this many samples')
cmd:option('-minhessian', 0.02, 'min hessian to avoid extreme speed up')
cmd:option('-maxhessian', 500, 'max hessian to avoid extreme slow down')

-- for conv models:
cmd:option('-kernelsize', 9, 'size of convolutional kernels')

-- logging:
cmd:option('-datafile', 'http://torch7.s3-website-us-east-1.amazonaws.com/data/tr-berkeley-N5K-M56x56-lcn.ascii', 'Dataset URL')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')
cmd:option('-v', false, 'be verbose')
cmd:option('-display', true, 'display stuff')
cmd:option('-wcar', '', 'additional flag to differentiate this run')
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

if params.display then
   displayData(dataset, 100, 10, 2)
end

----------------------------------------------------------------------
-- create model
--
dofile '2_models.lua'

----------------------------------------------------------------------
-- train model
--
dofile '3_train.lua'
