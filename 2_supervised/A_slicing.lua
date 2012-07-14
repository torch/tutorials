----------------------------------------------------------------------
-- slicing.lua
-- 
-- This script demonstrates tensor slicing / manipulation.

-- To run this script, simply do:
-- torch slicing.lua
-- and then press 'y' or 'return' at each step, to keep going.

-- little function to pause execution, and request user input
function next()
   local answer = nil
   while answer ~= '' and answer ~= 'y' and answer ~= 'Y' and neverstall ~= true do
      io.write("continue ([y]/n/!)? ")
      io.flush()
      answer=io.read()
      if answer == '!' then
         neverstall = true
      end
      if answer == 'n' then
         print('exiting...')
         os.exit()
      end
   end
   print ''
end

print '----------------------------------------------------------------------'
print 'creating a few tensors'

t1 = torch.range(1,75):resize(3,5,5)
print 't1 = torch.range(1,75):resize(3,5,5)'
print 't1 = '
print(t1)

t2 = torch.range(1,25):resize(5,5)
print 't2 = torch.range(1,25):resize(5,5)'
print 't2 = '
print(t2)

print 'done.'
print ''

next()
print '----------------------------------------------------------------------'
print 'the most basic slicing is done using the [] operator'
print ''

print 't1 ='
print( t1 )

print 't1[2] ='
print( t1[2] )

next()
print '----------------------------------------------------------------------'
print 't1_1 is a view in the existing t1 tensor: changing the values'
print 'in t1_1 directly affects t1:'
print ''

t1[2]:fill(7)
print 't1[2]:fill(7)'

print 't1[2] ='
print( t1[2] )

print 't1 ='
print( t1 )

next()
print '----------------------------------------------------------------------'
print 'more complex slicing can be done using the [{}] operator'
print 'this operator lets you specify one list/number per dimension'
print 'for example, t2 is a 2-dimensional tensor, therefore'
print 'we should pass 2 lists/numbers to the [{}] operator:'
print ''

t2_slice1 = t2[{ {},2 }]
t2_slice2 = t2[{ 2,{} }]      -- equivalent to t2[2]
t2_slice3 = t2[{ {2},{} }]
t2_slice4 = t2[{ {1,3},{3,4} }]
t2_slice5 = t2[{ {3},{4} }]
t2_slice6 = t2[{ 3,4 }]

print 't2 = '
print(t2)

print 't2[{ {},2 }] ='
print(t2_slice1)

print 't2[{ 2,{} }] ='
print(t2_slice2)

print 't2[{ {2},{} }] ='
print(t2_slice3)

print 't2[{ {1,3},{3,4} }] ='
print(t2_slice4)

print 't2[{ {3},{4} }] ='
print(t2_slice5)

print 't2[{ 3,4 }] ='
print(t2_slice6)

next()
print '----------------------------------------------------------------------'
print 'negative indexes can also be used:'
print ''

t2_slice7 = t2[{ {},{2,-2} }]
t2_slice8 = t2[{ -1,-1 }]

print 't2[{ {},{2,-2} }] ='
print(t2_slice7)

print 't2[{ -1,-1 }] ='
print(t2_slice8)

next()
print '----------------------------------------------------------------------'
print 'in basic Lua, the = operator cannot be overloaded (that speeds up the language parser'
print 'a lot...), but you can use the [{}] operator to copy tensors, and subtensors:'
print ''

print 't3 = torch.Tensor(5)'
print 't3[{}] = t2[{ {},1 }]'

t3 = torch.Tensor(5)
t3[{}] = t2[{ {},1 }]

print 't3 ='
print(t3)

next()
print '----------------------------------------------------------------------'
print 'if you need to slice arbitrary subtensors, you will need to do it in steps:'
print ''

t4 = torch.Tensor(5,2)
t4[{ {},1 }] = t2[{ {},2 }]
t4[{ {},2 }] = t2[{ {},5 }]

print [[
t4 = torch.Tensor(5,2)
t4[{ {},1 }] = t2[{ {},2 }]
t4[{ {},2 }] = t2[{ {},5 }]
]]

print 't4 ='
print(t4)






