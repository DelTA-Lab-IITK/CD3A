--------------------------------------------------------
-- Torch Implementation of Curriculum based Dropout Discriminator for Domain Adaptation(CD3A) BMVC 2019
--- Written By Vinod Kumar Kurmi (vinodkumarkurmi@gmail.com)
require 'cutorch'
require 'cunn'
require 'optim'
require 'gnuplot'
require 'loadcaffe'
require 'image';
require 'torch';
require 'nn';
require 'xlua'
require 'loadcaffe'
require 'cudnn'
require 'misc/nnlr/nnlr' --- for layer wise learnig rate
----------------------------------------------------
local c = require 'trepl.colorize'
local data_tm = torch.Timer()
----------------------------------------------------
opt = {
    manual_seed=2,          -- Seed
	batchSize = 64,         -- batch Size
	Test_batchSize = 64,     -- Test time batch size, it may change after the last epoch of test data
	start_Batch_IndexTest=1, --batch index at time of testing
	loadSize = 256,         -- resize the loaded image to loadsize maintaining aspect ratio. -- see donkey_folder.lua
	fineSize = 227,         -- size of random crops
	nc = 3,                 -- # of channels in input
	nThreads = 1,           -- #  of data loading threads to use
	gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
	save='logs/',            -- Saving the logs of trainining
	net1_freeze='yes',      -- For not updating the first 3 Conv layer
	--momentum
	number_of_testclass=31,  -- number of class in test dataset, in general it is =source class but we can use less class also.
    lamda=0.5,                -- Lamda value for gradeint reversal value.(fix)
	momentum=0.9,
	baseLearningRate=0.0002,
	max_epoch=10000,
	gamma=0.001,   -- for inverse policy :  base_lr * (1 + gamma * iter) ^ (- power)
	power=0.75,    -- for inverse policy :  base_lr * (1 + gamma * iter) ^ (- power)
	max_epoch_grl=10000, -- For progress in process , calculate the lamda for grl
	alpha=10,  -- LR schdular (2nd way)
	num_of_iter=1,
	dropout_no=0.5
}
	cutorch.manualSeed(opt.manual_seed)
	torch.manualSeed(opt.manual_seed)



--=====================Tuning Parameters===================================
	local prev_accuracy=0
	batchSize =opt.batchSize
	opt.save=opt.save .. 'batchsize_' .. opt.batchSize
	torch.setnumthreads(1)
	torch.setdefaulttensortype('torch.FloatTensor')
--==============Ploting Fuction=============================================================================
	confusion = optim.ConfusionMatrix({'letter_tray','paper_notebook','printer','bike_helmet','desk_lamp','mobile_phone',
		'desk_chair','pen','phone','headphones','ring_binder','tape_dispenser','bookcase','back_pack','laptop_computer','stapler',
		'ruler','mouse','projector','trash_can','monitor','file_cabinet','speaker','punchers','desktop_computer','bottle',
		'mug','keyboard','scissors','bike','calculator'})

	print('Will save at '..opt.save)
	paths.mkdir(opt.save)
	testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
	testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
	testLogger.showPlot = false
	errorlog = optim.Logger(paths.concat(opt.save, 'error.log'))
	errorlog:setNames{'% Training Error (train set)', '% Testing Error(test set)'}
	errorlog.showPlot = false

	errorlog_dis = optim.Logger(paths.concat(opt.save, 'error_dis.log'))
	errorlog_dis:setNames{' Domain Error (source)', '  Domain Error (Target)'}
	errorlog_dis.showPlot = false
--==========================================================================================================
----------------------------------------------------
--Path Initilication
	local prototxt_name = 'pretrained_network/alexnet/deploy.prototxt'
    local binary_name = 'pretrained_network/alexnet/bvlc_alexnet.caffemodel'
	local net_orignal = loadcaffe.load(prototxt_name, binary_name,'cudnn');
	print(' net_orignal', net_orignal)

---------------------------------------------------------
	-- create Train data loader
	local DataLoader = paths.dofile('data/data.lua')
	local data = DataLoader.new(opt.nThreads, opt)
	print("Train Dataset Size: ", data:size())

	-- create Val data loader
	local DataLoaderVal = paths.dofile('data/data_target.lua')
	local dataVal = DataLoaderVal.new(opt.nThreads, opt)
	print("Val Dataset Size: ", dataVal:size())

	-- create Test data loader
	local DataLoaderTest = paths.dofile('data_test/data.lua')
	local dataTest = DataLoaderTest.new(0, opt)
	print("test new Dataset Size: ", dataTest:size())

----------------------------------------------------
	--===FUNCTIONS==============

	local function uti(filename)
	   local net = torch.load(filename)
	  net:apply(function(m) if m.weight then
		 m.gradWeight = m.weight:clone():zero();
		 m.gradBias = m.bias:clone():zero(); end end)
	   return net
	end


	local function check_accuracy(scores, targets)
		local num_test = (#targets)[1]
		local no_correct = 0
		local confidences, indices = torch.sort(scores, true)
		local predicted_classes = indices[{{},{1}}]:long()
		targets = targets:long()
		no_correct = no_correct + ((torch.squeeze(predicted_classes):eq(targets)):sum())
		local accuracy = no_correct / num_test
		return accuracy
	end


	function check_accuracyTest(scores, targets)
        local num_test = (#targets)[1]
        local no_correct = 0
        local confidences, indices = torch.sort(scores, true)
        local predicted_classes = indices[{{},{1}}]:long()
        targets = targets:long()
        if num_test==1 then
            no_correct = no_correct + ((predicted_classes:eq(targets)):sum())
        else
            no_correct = no_correct + ((torch.squeeze(predicted_classes):eq(targets)):sum())
        end
        local accuracy = no_correct
        return accuracy
    end

--=======================Model==========================================================================
	--------------Load Alexnet Pretrained Netwok
	local model = nn.Sequential()
	--------Map table for two stream input(one for source data another for target data)---------------
	local net1= nn.MapTable()
	local net2= nn.MapTable()
	local net3= nn.MapTable()
	local net4= nn.MapTable()
	local netB= nn.MapTable()
	local netD= nn.MapTable()

	local net11= nn.Sequential()
	local net22= nn.Sequential()
	local net33= nn.Sequential()
	local net44= nn.Sequential()
	local netDD= nn.Sequential()
	local netBB= nn.Sequential()

	-- Layer by layer copy from the pretrained Alexnet  Netwrok
	for i, module in ipairs( net_orignal.modules) do
		 if(i<11) then
			if(i==1) then
				module:setMode(1,1,1)  -- For making determinstic the cudnn convolution layers
		        net11:add(module):learningRate('weight', 1)  -- Layer wise learning rate
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --conv1
			elseif (i==4) then
				max1 = nn.SpatialMaxPooling(3, 3, 2,2)     --Replace cudnn.maxpooling-->nn.maxpooling for deterministic response
		       	 net11:add(max1)

			elseif (i==5) then
				module:setMode(1,1,1) -- For making determinstic the cudnn convolution layers
		        net11:add(module):learningRate('weight', 1)
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --conv2
			elseif (i==8) then
				max2 = nn.SpatialMaxPooling(3, 3, 2,2)   --Replace cudnn.maxpooling-->nn.maxpooling for deterministic response
		       	 net11:add(max2)

			elseif (i==9) then
				module:setMode(1,1,1) -- For making determinstic the cudnn convolution layers
		        net11:add(module):learningRate('weight', 1)
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --conv3

			else
                         if (i ~= 4 and i ~=8) then
					net11:add(module)
				end
			end
		elseif (i>10 and i<17) then
			if(i==11) then
				module:setMode(1,1,1)

		        net22:add(module):learningRate('weight', 1)
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --conv4

			elseif (i==13) then
				module:setMode(1,1,1)

		        net22:add(module):learningRate('weight', 1)
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --conv5
			elseif (i==15) then
				max3 = nn.SpatialMaxPooling(3, 3, 2,2)   --Replace cudnn.maxpooling-->nn.maxpooling for deterministic response
		       	 net22:add(max3)

			else
				 if (i ~= 15) then
				net22:add(module)
				end

			end
		else
			if(i==17) then
		        net33:add(module):learningRate('weight', 1)
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --FC6
			elseif (i==20) then
		        net33:add(module):learningRate('weight', 1)
						  :learningRate('bias', 2)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --FC7
			elseif (i==23) then
		        net33:add(module):learningRate('weight', 0)
						  :learningRate('bias', 0)
						  :weightDecay('weight', 0)
						  :weightDecay('bias', 0) --FC8
			else
				net33:add(module)
			end
		 end
	end

	-- print('net33',net33.modules)
	net33:remove(#net33)    --removed softmax
	net33:remove(#net33)    --removed FC8

      -- Bottlenec Network------
	netBB:add(nn.Linear( 4096, 256)):learningRate('weight', 10)
						  :learningRate('bias', 20)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0)
	netBB:add(nn.ReLU(true))

      -- Gradient Reversal Domain classifier Network
	module = nn.GradientReversal(lambda)
	netDD:add(module)
	netDD:add( nn.Linear( 256, 1024)):learningRate('weight', 10)
						  :learningRate('bias', 20)
	netDD:add(nn.ReLU(true))
	netDD:add(nn.Dropout(opt.dropout_no))
	netDD:add( nn.Linear( 1024, 1024)):learningRate('weight', 10)
						  :learningRate('bias', 20)
	netDD:add(nn.ReLU(true))
	netDD:add(nn.Dropout(opt.dropout_no))
	netDD:add( nn.Linear( 1024, 1)):learningRate('weight', 10)
						  :learningRate('bias', 20)
	netDD:add(nn.Sigmoid())  -- removed this layer if we are using the nn.CrossEntropyCriterion()


     -- Classifier Network------------------
	net44:add( nn.Linear( 256, 31)):learningRate('weight', 10)
						  :learningRate('bias', 20)
						  :weightDecay('weight', 1)
						  :weightDecay('bias', 0) --FC6
	net44:add(nn.LogSoftMax())
    -- Map Tabel for two input----
	net1:add(net11)
	net2:add(net22)
	net3:add(net33)
	net4:add(net44)
	netB:add(netBB)
	netD:add(netDD)

      --Initially Lamda set =0
	module:setLambda(0)

--============ Criterion=================
	local criterion = nn.ClassNLLCriterion()
	local criterionNLL = nn.ClassNLLCriterion()
	local criterionNLL_parallel = nn.ParallelCriterion():add(criterionNLL):add(criterionNLL)
	-- local criterionMSE = nn.MSECriterion()
	-- local criterionMSE_parallel = nn.ParallelCriterion():add(criterionMSE,1):add(criterionMSE,1)
	local criterionBCE = nn.BCECriterion()
	local criterionBCE_sg = nn.BCECriterion()
	local criterionBCE_tg = nn.BCECriterion()
	local criterionBCE_parallel = nn.ParallelCriterion():add(criterionBCE,1):add(criterionBCE,1)
--==========================================

-----------------------------------------------------------------------------------------------
	if opt.gpu >=0 then
		net1:cuda()
		net2:cuda()
		net3:cuda()
		netB:cuda()
		net4:cuda()
		netD:cuda()
		criterion:cuda()
		criterionNLL_parallel:cuda()
		-- criterionMSE_parallel:cuda()
		criterionBCE_parallel:cuda()
		criterionBCE_sg:cuda()
		criterionBCE_tg:cuda()
	 end

--=== Different Learning rate for weigth and bias
	local temp_baseWeightDecay=0.001  --no meaningin my case
	local learningRates_Net1, weightDecays_Net1 = net1:getOptimConfig(opt.baseLearningRate,temp_baseWeightDecay)
	local learningRates_Net2, weightDecays_Net2 = net2:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_Net3, weightDecays_Net3 = net3:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_NetB, weightDecays_NetB = netB:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_NetD, weightDecays_NetD = netD:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
	local learningRates_Net4, weightDecays_Net4 = net4:getOptimConfig(opt.baseLearningRate, temp_baseWeightDecay)
--===========Parameters===================================
	local parameters1, gradParameters1 = net1:getParameters()
	local parameters2, gradParameters2 = net2:getParameters()
	local parameters3, gradParameters3 = net3:getParameters()
	local parameters4, gradParameters4 = net4:getParameters()
	local parametersB, gradParametersB = netB:getParameters()
	local parametersD, gradParametersD = netD:getParameters()
--============================================================

	local method = 'xavier'
	net4 = require('misc/weight-init')(net4, method)
	netB = require('misc/weight-init')(netB, method)
	netD = require('misc/weight-init')(netD, method)
----------------------------------------------------------------------------------------------------
	print('=> New Model')
	print(model)
	print('net1', net1)
	print('net2', net2)
	print('net3', net3)
	print('netB', netB)
	print('net4', net4)
	print(criterion)
	collectgarbage()
	local updated_learningrate=opt.baseLearningRate

--===================Training Fuctions======================================

function train()
	net1:training()
	net2:training()
	net3:training()
	net4:training()
	netB:training()
	netD:training()
	epoch = epoch or 1
	if(epoch>1) then
	print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
		local p=epoch/opt.max_epoch_grl
		local baseWeightDecay = torch.pow((1 +  epoch * opt.gamma), (-1  * opt.power)) -- need to chanage
            updated_learningrate=opt.baseLearningRate*baseWeightDecay
		print('Learnig Rate',updated_learningrate)
		--lamda=(2*torch.pow(1+torch.exp(-10*p),-1))-1
		print('opt.num_of_iter',opt.num_of_iter)
		module:setLambda(opt.lamda)
		if(epoch>10) then
			opt.num_of_iter=epoch/10
		else
			opt.num_of_iter=1
		end
		print('opt.num_of_iter',opt.num_of_iter)

	end
	local avg_loss=0
	local avg_acc=0
	local count =0
	local avg_s_d=0
	local avg_t_d=0
	for i = 1, data:size(), opt.batchSize do
		-----Classifier Network-------------------
		data_tm:reset(); data_tm:resume()
		local batchInputs_source,label = data:getBatch()
		local batchInputs_target = dataVal:getBatch()
		local SlabelDomain=torch.Tensor(opt.batchSize):fill(0)
		local TlabelDomain=torch.Tensor(opt.batchSize):fill(1)
		if opt.gpu >=0 then
			label=label:cuda()
			SlabelDomain=SlabelDomain:cuda()
			TlabelDomain=TlabelDomain:cuda()
			batchInputs_source=batchInputs_source:cuda()
			batchInputs_target=batchInputs_target:cuda()
		end
--------------------------------------------------------------------------
	-- forwardNetwork
		local outputs1 = net1:forward({batchInputs_source,batchInputs_target})
		local outputs2 = net2:forward(outputs1)
		local outputs3 = net3:forward(outputs2)
		local outputsB = netB:forward(outputs3)
		local outputs4 = net4:forward(outputsB)


		local err = criterion:forward(outputs4[1], label)
		local errd_s=0
		local errd_t=0



		gradParametersD:zero()
        for DomainI=1,opt.num_of_iter do
	        local outputsD = netD:forward(outputsB)
	        local err_sg=criterionBCE_sg:forward(outputsD[1], SlabelDomain)
	        local err_tg=criterionBCE_tg:forward(outputsD[2], TlabelDomain)
	        errd_s=errd_s+err_sg
	        errd_t=errd_t+err_tg
			-- local errDomain = criterionBCE_parallel:forward(outputsD, {SlabelDomain,TlabelDomain})
			-- local dgradOutputsDomain = criterionBCE_parallel:backward(outputsD, {SlabelDomain,TlabelDomain})
			local derr_sg=criterionBCE_sg:backward(outputsD[1], SlabelDomain)
	        local derr_tg=criterionBCE_tg:backward(outputsD[2], TlabelDomain) 	 -- classification loss grad
	        dgradOutputs_modD  = netD:backward(outputsB, {derr_sg,derr_tg})
        end
        errd_s=errd_s/opt.num_of_iter
        errd_t=errd_t/opt.num_of_iter
	---- Optimization Domain Confusion Branch -------
		local feval_netD = function(x)
			return err, gradParametersD
		end
		optim.sgd(feval_netD, parametersD, {
							       learningRates = learningRates_NetD,
							       weightDecays = weightDecays_NetD,
							       learningRate = updated_learningrate,
							       momentum = opt.momentum,
							      })
--------------------------------------------------------------------------
-- backward Network
		gradParameters4:zero()
		gradParametersB:zero()
		gradParameters3:zero()
		gradParameters2:zero()
		gradParameters1:zero()

		local dgradOutputsS=torch.CudaTensor()    --Declaration of dgradOutputsS for source class
		dgradOutputsS:resize(outputs4[1]:size())
		dgradOutputsS:zero()
		dgradOutputsS = criterion:backward(outputs4[1], label)

     	local zeros = torch.CudaTensor()     -- Zero gradient for Target data Classification(we dont have target label)
		zeros:resize(dgradOutputsS:size())
		zeros:zero()
		dgradOutputs={dgradOutputsS, zeros}

	---- Optimization Net4-------
		local feval_net4 = function(x)
		dgradOutputs_mod4 = net4:backward(outputsB, dgradOutputs)
			return err, gradParameters4
		end
		optim.sgd(feval_net4, parameters4, {
							       learningRates = learningRates_Net4,
							       weightDecays = weightDecays_Net4,
							       learningRate = updated_learningrate,
							       momentum = opt.momentum,
							      })


	---- Optimization netB(bottleneck_ Branch -------
		local total_grad={}
		total_grad[1] = dgradOutputs_mod4[1]+ dgradOutputs_modD[1]
		total_grad[2] = dgradOutputs_mod4[2]+ dgradOutputs_modD[2]
		feval_netB = function(x)
			dgradOutputs_modB   = netB:backward(outputs3,total_grad)
			return err, gradParametersB
		end
		optim.sgd(feval_netB, parametersB, {
							       learningRates = learningRates_NetB,
							       weightDecays = weightDecays_NetB,
							       learningRate = updated_learningrate,
							       momentum = opt.momentum,
							      })

	---- Optimization net3(FC6,FC7) Branch -------
		gradParameters3:zero()
		local feval_net3 = function(x)
		dgradOutputs_mod3= net3:backward(outputs2,dgradOutputs_modB)
			return  err, gradParameters3
		end
		optim.sgd(feval_net3, parameters3, {
							       learningRates = learningRates_Net3,
							       weightDecays = weightDecays_Net3,
							       learningRate = updated_learningrate,
							       momentum = opt.momentum,
							      })
	---- Optimization net2(Conv4 -Pool5) Branch -------
		gradParameters2:zero()
		local feval_net2 = function(x)
		dgradOutputs_mod2= net2:backward(outputs1,dgradOutputs_mod3)
			return  err, gradParameters2
		end
		optim.sgd(feval_net2, parameters2, {
							       learningRates = learningRates_Net2,
							       weightDecays = weightDecays_Net2,
							       learningRate = updated_learningrate,
							       momentum = opt.momentum,
							      })
		---- if required then Net1 optimization----
		if opt.net1_freeze =='no' then
			gradParameters1:zero()
			feval_net1 = function(x)
				model.modules[1]:backward((batchInputs_source),dgradOutputs_mod2)
				return  gradOutputs_mod1, gradParameters1
				end
			optim.sgd(feval_net1, parameters1, {
							       learningRates = learningRates_Net1,
							       weightDecays = weightDecays_Net1,
							       learningRate = updated_learningrate,
							       momentum = opt.momentum,
							      })
		end

		local train_acc = check_accuracy(outputs4[1], label)
		avg_loss=avg_loss+err
		avg_acc=avg_acc+train_acc
		train_acc =nil
		err=nil

		avg_s_d=avg_s_d+errd_s
		avg_t_d=avg_t_d+errd_t
		batchInputs_source=nil
		label=nil
		SlabelDomain=nil
		TlabelDomain=nil
		batchInputs_target=nil
		outputs1=nil
		outputs2=nil
		outputs3=nil
		outputs4=nil
		outputs=nil
		dgradOutputsS=nil
		zeros=nil
		count=count+1
	end
  epoch = epoch + 1

return (avg_loss)/count,avg_acc/count,avg_s_d/count,avg_t_d/count
end


--==============================Testing===================================================================
function test()
	-- disable flips, dropouts and batch normalization
      net1:evaluate()
      net2:evaluate()
      net3:evaluate()
      netB:evaluate()
      net4:evaluate()
      netD:evaluate()
	local err_val=0
	local avg_test_acc=0
	local count=0
	opt.Test_batchSize=opt.batchSize
	for i = 1,dataTest:size(), opt.Test_batchSize do
		data_tm:reset(); data_tm:resume()
		opt.start_Batch_IndexTest=i
		local batchInputs_test,validLabel = dataTest:getBatch(opt.start_Batch_IndexTest,dataTest:size())
		if opt.gpu >=0 then
			batchInputs_test=batchInputs_test:cuda()
			validLabel=validLabel:cuda()
		end
		local outputs1 = net1:forward({batchInputs_test:cuda()})
		local outputs2 = net2:forward(outputs1)
		local outputs3 = net3:forward(outputs2)
		local outputsB = netB:forward(outputs3)
		local outputs =  net4:forward(outputsB)
		confusion:batchAdd(outputs[1], validLabel)
		err_val = err_val+ criterion:forward(outputs[1],validLabel)  -- Classification Loss
		count=count+1
	   local test_batch_acc = check_accuracyTest(outputs[1], validLabel)
	   avg_test_acc=avg_test_acc+test_batch_acc
	   test_batch_acc=nil

	end
	confusion:updateValids()
	test_accuracy=confusion.totalValid
	if not testLogger then
		confusion:zero()
	end
	 return err_val/count, test_accuracy,avg_test_acc/dataTest:size()
end

function save_html(train_acc,test_acc,train_err,test_err,errdS,errdT)
	if testLogger then
		paths.mkdir(opt.save)
		testLogger:add{train_acc, test_acc}
		testLogger:style{'-','-'}
		-- testLogger:plot()
		errorlog:add{train_err, test_err}
		errorlog:style{'-','-'}
		-- errorlog:plot()
		errorlog_dis:add{errdS,errdT}
		errorlog_dis:style{'-','-'}
		if paths.filep(opt.save..'/test.log.eps') then
			local base64im
			do
				os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/test.base64')
				if f then base64im = f:read'*all' end
			end
			local base64im_error
			do
				os.execute(('convert -density 200 %s/error.log.eps %s/error.png'):format(opt.save,opt.save))
				os.execute(('openssl base64 -in %s/error.png -out %s/error.base64'):format(opt.save,opt.save))
				local f = io.open(opt.save..'/error.base64')
				if f then base64im_error = f:read'*all' end
			end
			local file = io.open(opt.save..'/report.html','w')
			file:write('<h5>Training data size:  '..data:size()..'\n')
			file:write('<h5>Validation data size:  '..dataTest:size()..'\n')
			file:write('<h5>batchSize:  '..batchSize..'\n')
			file:write('<h5>Network upto conv3 is freeze:   '..opt.net1_freeze..'\n')
			file:write('<h5>Base Learning Rate:  '..opt.baseLearningRate..'\n')
			file:write('<h5>momentum:  '..opt.momentum..'\n')
			file:write('<h5>Seed :  '..opt.manual_seed..'\n')
			file:write('<h5>lamda :  '..opt.lamda..'\n')
			file:write('<h5>number of test Class :  '..opt.number_of_testclass..'\n')
			file:write'</table><pre>\n'
			file:write(tostring(confusion)..'\n')
			file:write(tostring(net4)..'\n')
			file:write'</pre></body></html>'
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im))
			file:write(([[
			<!DOCTYPE html>
			<html>
			<body>
			<title>%s - %s</title>
			<img src="data:image/png;base64,%s">
			<table>
			]]):format(opt.save,epoch,base64im_error))

			file:close()
		end
		confusion:zero()
	end

		--print('epoch',epoch)
          if prev_accuracy< test_acc then
		print('Model is saving')
		collectgarbage()
	      net1:clearState()
	      net2:clearState()
	      net3:clearState()
	      netB:clearState()
	      net4:clearState()
	      netD:clearState()

		torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'net1_' .. epoch .. '.t7'),net1) -- defined in util.lua
		torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'net2_' .. epoch .. '.t7'),net2)
		torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'net3_' .. epoch .. '.t7'),net3)
		torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'netB_' .. epoch .. '.t7'),netB)
		torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'net4_' .. epoch .. '.t7'),net4)
		torch.save(paths.concat(opt.save, 'Accuracy' .. test_acc .. 'netD_' .. epoch .. '.t7'),netD)
		print('Model is Saved')
		prev_accuracy=test_acc
		end
end

for i=1,opt.max_epoch do
	train_loss,train_acc,errdS,errdT=train()
	print('Train_acc',train_acc,'Train_loss',train_loss)
	collectgarbage()
	test_loss,test_acc,test_acc_2=test()
	print('test_acc',test_acc, 'test_acc_2',test_acc_2,'Test_loss',test_loss)
	save_html(train_acc,test_acc,train_loss,test_loss,errdS,errdT)
	collectgarbage()
end



