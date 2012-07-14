
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
-- Simple example, doing decoding and inference
--
do
   -- define graph
   nNodes = 10
   nStates = 2
   adjacency = gm.adjacency.full(nNodes)
   g = gm.graph{adjacency=adjacency, nStates=nStates, maxIter=10, verbose=true}

   -- unary potentials
   nodePot = tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,1},
                    {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

   -- joint potentials
   edgePot = tensor(g.nEdges,nStates,nStates)
   basic = tensor{{2,1}, {1,2}}
   for e = 1,g.nEdges do
      edgePot[e] = basic
   end

   -- set potentials
   g:setPotentials(nodePot,edgePot)

   -- exact inference
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   --print('<gm.testme> edge beliefs:')
   --print(edgeBel)
   print('<gm.testme> log(Z):')
   print(logZ)

   -- bp inference
   local bp = g:decode('bp')
   print()
   print('<gm.testme> optimal config with belief propagation:')
   print(bp)

   local nodeBel,edgeBel,logZ = g:infer('bp')
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   --print('<gm.testme> edge beliefs:')
   --print(edgeBel)
   print('<gm.testme> log(Z):')
   print(logZ)
end
