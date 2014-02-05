
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
   -- define graph:
   nNodes = 10
   nStates = 2
   adjacency = torch.zeros(nNodes,nNodes)
   for i = 1,nNodes-1 do
      adjacency[i][i+1] = 1
      adjacency[i+1][i] = 1
   end
   g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, type='mrf', verbose=true}

   -- define training set:
   nInstances = 100
   Y = tensor(nInstances,nNodes)
   for i = 1,nInstances do
      -- each entry is either 1 or 2, with a probability that
      -- increases with the node index
      for n = 1,nNodes do
         Y[i][n] = torch.bernoulli((n-1)/(nNodes-1)) + 1
      end
      -- create correlation between last two nodes
      Y[i][nNodes-1] = Y[i][nNodes]
   end

   -- NOTE: the 10 training nodes in Y have probability 0, 1/9, ... , 9/9 to be equal
   -- to 2. The node beliefs obtained after training should show that.

   -- tie node potentials to parameter vector
   -- NOTE: we allocate one parameter per node, to properly model
   -- the probability of each node
   nodeMap = zeros(nNodes,nStates)
   for n = 1,nNodes do
      nodeMap[{ n,1 }] = n
   end

   -- tie edge potentials to parameter vector
   -- NOTE: we allocate parameters globally, i.e. parameters model
   -- pairwise relations globally
   nEdges = g.edgeEnds:size(1)
   edgeMap = zeros(nEdges,nStates,nStates)
   edgeMap[{ {},1,1 }] = nNodes+1
   edgeMap[{ {},2,2 }] = nNodes+2
   edgeMap[{ {},1,2 }] = nNodes+3

   -- initialize parameters
   g:initParameters(nodeMap,edgeMap)
   
   -- estimate nll:
   require 'optim'
   optim.lbfgs(function()
      local f,grad = g:nll('exact',Y)
      print('LBFGS – objective = ', f)
      return f,grad
   end, g.w, {maxIter=100, lineSearch=optim.lswolfe})

   -- gen final potentials
   g:makePotentials()

   -- exact decoding:
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   -- exact inference:
   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs (prob that node=2)')
   print(nodeBel[{ {},2 }])
   print('<gm.testme> edge beliefs (prob that node1=2 & node2=2)')
   print(edgeBel[{ {},2,2 }])

   -- sample from model:
   local samples = g:sample('exact',5)
   print('<gm.testme> 5 samples from model:')
   print(samples)

   local samples = g:sample('gibbs',5)
   print('<gm.testme> 5 samples from model (Gibbs):')
   print(samples)
end
