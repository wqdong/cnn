% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
% TODO: FORWARD CODE
expIn = exp(input);
expSum = sum(exp(input));
for i = 1:batch_size
   output(:,i) = exp(input(:,i))/expSum(i);
end

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    dy_dx = zeros(num_classes);
    for i = 1:batch_size
        for j = 1:num_classes
            for k = 1:num_classes
                if j == k
                    dy_dx(j,k) = exp(input(k,i))*(expSum(i)-exp(input(k,i)))/expSum(i)^2;
                else
                    dy_dx(j,k) = -exp(input(k,i))*exp(input(j,i))/expSum(i)^2;
                end
            end
        end
        dv_input(:,i) = dy_dx*dv_output(:,i);
    end
end
