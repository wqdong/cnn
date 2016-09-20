function [model, loss, accuracy] = train(model,input,label,params,numIters)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.
addpath pcode

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .01; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

loss = [];
accuracy = [];
for i = 1:numIters
	% TODO: Training code
    index = randperm(size(input,4),batch_size);
    batch = input(:,:,:,index);
    batch_label = label(index);
    [output,activations] = inference(model,batch);
    [temp,dv] = loss_crossentropy(output, batch_label, [], 1);
    loss = [loss; temp(:)];
    grad = calc_gradient(model, batch, activations, dv);
    update_params = struct('learning_rate',lr,'weight_decay',wd);
    updated_model = update_weights(model,grad,update_params);
    model = updated_model;
    [~,label_pred] = max(output);
    
    correctNum = 0.0;
    for j = 1:batch_size
        if(label_pred(j)==batch_label(j))
            correctNum = correctNum + 1;
        end
    end
    
    %correctNum = sum(label_pred == batch_label);
    temp = correctNum/batch_size;
    accuracy = [accuracy; temp(:)];
end

if numIters == 1 
    disp('test 10000: ')
    disp(loss)
    disp(accuracy)
end