% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
loss = 0.0;
labelMat = zeros(size(input));
for i = 1:size(labels,1)
    labelMat(labels(i),i) = 1;
    loss = loss - dot(labelMat(:,i),log(input(:,i)));
end
loss = loss/size(input,2);

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    for i = 1:size(labels,1)
        dv_input(:,i) = -labelMat(:,i)./input(:,i)/size(input,2);
    end
end
