
-- graphical model lib
require 'gm'

-- shortcuts
local tensor = torch.Tensor
local zeros = torch.zeros
local ones = torch.ones
local randn = torch.randn
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp
local floor = torch.floor
local ceil = math.ceil
local uniform = torch.uniform

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Example of how to train a CRF for a simple segmentation task
--
do
   -- make training data
   sample = torch.load('X.t7')
   nRows,nCols = sample:size(1),sample:size(2)
   nNodes = nRows*nCols
   nStates = 2
   nInstances = 100

   -- make labels (MAP):
   y = tensor(nInstances,nRows*nCols)
   for i = 1,nInstances do
      y[i] = sample
   end
   y = y + 1

   -- make noisy training data:
   X = tensor(nInstances,1,nRows*nCols)
   for i = 1,nInstances do
      X[i] = sample
   end
   X = X + randn(X:size())/2

   -- display a couple of input examples
   if itorch then
      print('training examples visualization:')
      itorch.image({
	    X[1]:reshape(32,32),
	    X[2]:reshape(32,32),
	    X[3]:reshape(32,32),
	    X[4]:reshape(32,32)
      })
   end

   -- define adjacency matrix (4-connexity lattice)
   local adj = gm.adjacency.lattice2d(nRows,nCols,4)

   -- create graph
   g = gm.graph{adjacency=adj, nStates=nStates, verbose=true, type='crf', maxIter=10}

   -- create node features (normalized X and a bias)
   Xnode = tensor(nInstances,2,nNodes)
   Xnode[{ {},1 }] = 1 -- bias
   -- normalize features:
   nFeatures = X:size(2)
   for f = 1,nFeatures do
      local Xf = X[{ {},f }]
      local mu = Xf:mean()
      local sigma = Xf:std()
      Xf:add(-mu):div(sigma)
   end
   Xnode[{ {},2 }] = X -- features (simple normalized grayscale)
   nNodeFeatures = Xnode:size(2)

   -- tie node potentials to parameter vector
   nodeMap = zeros(nNodes,nStates,nNodeFeatures)
   for f = 1,nNodeFeatures do
      nodeMap[{ {},1,f }] = f
   end

   -- create edge features
   nEdges = g.edgeEnds:size(1)
   nEdgeFeatures = nNodeFeatures*2-1 -- sharing bias, but not grayscale features
   Xedge = zeros(nInstances,nEdgeFeatures,nEdges)
   for i = 1,nInstances do
      for e =1,nEdges do
         local n1 = g.edgeEnds[e][1]
         local n2 = g.edgeEnds[e][2]
         for f = 1,nNodeFeatures do
            -- get all features from node1
            Xedge[i][f][e] = Xnode[i][f][n1]
         end
         for f = 1,nNodeFeatures-1 do
            -- get all features from node1, except bias (shared)
            Xedge[i][nNodeFeatures+f][e] = Xnode[i][f+1][n2]
         end
      end
   end

   -- tie edge potentials to parameter vector
   local f = nodeMap:max()
   edgeMap = zeros(nEdges,nStates,nStates,nEdgeFeatures)
   for ef = 1,nEdgeFeatures do
      edgeMap[{ {},1,1,ef }] = f+ef
      edgeMap[{ {},2,2,ef }] = f+ef
   end

   -- initialize parameters
   g:initParameters(nodeMap,edgeMap)

   -- and train on 30 samples
   require 'optim'
   local sgdState = {
      learningRate = 1e-3,
      learningRateDecay = 1e-2,
      weightDecay = 1e-5,
   }
   for iter = 1,100 do
      -- SGD step:
      optim.sgd(function()
         -- random sample:
         local i = torch.random(1,nInstances)
         -- compute f+grad:
         local f,grad = g:nll('bp',y[i],Xnode[i],Xedge[i])
         -- verbose:
         print('SGD @ iteration ' .. iter .. ': objective = ', f)
         -- return f+grad:
         return f,grad
      end, 
      g.w, sgdState)
   end

   -- the model is trained, generate node/edge potentials, and test
   marginals = {}
   labelings = {}
   for i = 1,4 do
      g:makePotentials(Xnode[i],Xedge[i])
      nodeBel = g:infer('bp')
      labeling = g:decode('bp')
      table.insert(marginals,nodeBel[{ {},2 }]:reshape(nRows,nCols))
      table.insert(labelings,labeling:reshape(nRows,nCols))
   end

   -- display
   if itorch then
      print('marginals:')
      itorch.image(marginals)
   end
   for _,labeling in ipairs(labelings) do
      labeling:add(-1)
   end
   if itorch then
      print('labelings:')
      itorch.image(labelings)
   end
end
