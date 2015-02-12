
local data_verbose = false

function getdata(datafile, inputsize, std)
   local data = torch.load(datafile, 'ascii')
   local dataset ={}

   local std = std or 0.2
   local nsamples = data:size(1)
   local nrows = data:size(2)
   local ncols = data:size(3)

   function dataset:size()
      return nsamples
   end

   function dataset:selectPatch(nr,nc)
      local imageok = false
      if simdata_verbose then
         print('selectPatch')
      end
      while not imageok do
         --image index
         local i = math.ceil(torch.uniform(1e-12,nsamples))
         local im = data:select(1,i)
         -- select some patch for original that contains original + pos
         local ri = math.ceil(torch.uniform(1e-12,nrows-nr))
         local ci = math.ceil(torch.uniform(1e-12,ncols-nc))
         local patch = im:narrow(1,ri,nr)
         patch = patch:narrow(2,ci,nc)
         local patchstd = patch:std()
         if data_verbose then
            print('Image ' .. i .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
         end
         if patchstd > std then
            if data_verbose then
               print(patch:min(),patch:max())
            end
            return patch,i,im
         end
      end
   end

   local dsample = torch.Tensor(inputsize*inputsize)

   function dataset:conv()
      dsample = torch.Tensor(1,inputsize,inputsize)
   end

   setmetatable(dataset, {__index = function(self, index)
                                       local sample,i,im = self:selectPatch(inputsize, inputsize)
                                       dsample:copy(sample)
                                       return {dsample,dsample,im}
                                    end})
   return dataset
end

function getdatacam(inputsize, std)
   require 'camera'
   local frow = 60
   local fcol = 80
   local gs = 5
   local cam = image.Camera{width=fcol,height=frow}
   local dataset ={}
   local counter = 1

   local std = std or 0.2
   local nsamples = 10000
   local gfh = image.gaussian{width=gs,height=1,normalize=true}
   local gfv = image.gaussian{width=1,height=gs,normalize=true}
   local gf = image.gaussian{width=gs,height=gs,normalize=true}

   function dataset:size()
      return nsamples
   end


   local imsq = torch.Tensor()
   local lmnh = torch.Tensor()
   local lmn = torch.Tensor()
   local lmnsqh = torch.Tensor()
   local lmnsq = torch.Tensor()
   local lvar = torch.Tensor()
   local function lcn(im)
      local mn = im:mean()
      local std = im:std()
      if data_verbose then
         print('im',mn,std,im:min(),im:max())
      end
      im:add(-mn)
      im:div(std)
      if data_verbose then
         print('im',im:min(),im:max(),im:mean(), im:std())
      end

      imsq:resizeAs(im):copy(im):cmul(im)
      if data_verbose then
         print('imsq',imsq:min(),imsq:max())
      end

      torch.conv2(lmnh,im,gfh)
      torch.conv2(lmn,lmnh,gfv)
      if data_verbose then
         print('lmn',lmn:min(),lmn:max())
      end

      --local lmn = torch.conv2(im,gf)
      torch.conv2(lmnsqh,imsq,gfh)
      torch.conv2(lmnsq,lmnsqh,gfv)
      if data_verbose then         
         print('lmnsq',lmnsq:min(),lmnsq:max())
      end

      lvar:resizeAs(lmn):copy(lmn):cmul(lmn)
      lvar:mul(-1)
      lvar:add(lmnsq)
      if data_verbose then      
         print('2',lvar:min(),lvar:max())
      end

      lvar:apply(function (x) if x<0 then return 0 else return x end end)
      if data_verbose then
         print('2',lvar:min(),lvar:max())
      end

      local lstd = lvar
      lstd:sqrt()
      lstd:apply(function (x) if x<1 then return 1 else return x end end)
      if data_verbose then
         print('lstd',lstd:min(),lstd:max())
      end

      local shift = (gs+1)/2
      local nim = im:narrow(1,shift,im:size(1)-(gs-1)):narrow(2,shift,im:size(2)-(gs-1))
      nim:add(-1,lmn)
      nim:cdiv(lstd)
      if data_verbose then
         print('nim',nim:min(),nim:max())
      end

      return nim
   end

   function dataset:selectPatch(nr,nc)
      local imageok = false
      if simdata_verbose then
         print('selectPatch')
      end
      counter = counter + 1
      local imgray = image.rgb2y(cam:forward())

      local nim = lcn(imgray[1]:clone())
      while not imageok do

         -- select some patch for original that contains original + pos
         local ri = math.ceil(torch.uniform(1e-12,nim:size(1)-nr))
         local ci = math.ceil(torch.uniform(1e-12,nim:size(2)-nc))
         local patch = nim:narrow(1,ri,nr)
         patch = patch:narrow(2,ci,nc)
         local patchstd = patch:std()
         if data_verbose then
            print('Image ' .. 0 .. ' ri= ' .. ri .. ' ci= ' .. ci .. ' std= ' .. patchstd)
         end
         if patchstd > std then
            if data_verbose then
               print(patch:min(),patch:max())
            end
            return patch,i,nim
         end
      end
   end

   local dsample = torch.Tensor(inputsize*inputsize)
   setmetatable(dataset, {__index = function(self, index)
                                       local sample,i,im = self:selectPatch(inputsize, inputsize)
                                       dsample:copy(sample)
                                       return {dsample,dsample,im}
                                    end})
   return dataset
end

-- dataset, dataset=createDataset(....)
-- nsamples, how many samples to display from dataset
-- nrow, number of samples per row for displaying samples
-- zoom, zoom at which to draw dataset
function displayData(dataset, nsamples, nrow, zoom)
   require 'image'
   local nsamples = nsamples or 100
   local zoom = zoom or 1
   local nrow = nrow or 10

   cntr = 1
   local ex = {}
   for i=1,nsamples do
      local exx = dataset[1]
      ex[cntr] = exx[1]:clone():unfold(1,math.sqrt(exx[1]:size(1)),math.sqrt(exx[1]:size(1)))
      cntr = cntr + 1
   end
   if itorch then
      itorch.image(ex)
   else
      print('For visualization, run the script in itorch')
   end
end
