% Basic script to create a new network model
load_MNIST_data;
addpath pcode;
addpath layers;

l = [init_layer('conv',struct('filter_size',10,'filter_depth',1,'num_filters',10))
	init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',810,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);

train_params = struct('learning_rate',0.12,'weight_decay',0.0002,'batch_size',128);
test_all_params = struct('learning_rate',0.0,'weight_decay',0.0,'batch_size',10000);

loss_train_mat = [];
acc_train_mat = [];
loss_test_mat = [];
acc_test_mat = [];
tic;
for i = 1:15
    disp('iteration #');disp(i);
    %training
    [trained_model, loss_train, acc_train] = train(model,train_data,train_label,train_params,100);

    %testing
    [~,loss_test, acc_test] = train(trained_model,test_data,test_label,test_all_params,1);
    
    %update
    model = trained_model;
    train_params.learning_rate = train_params.learning_rate*0.95;
    
    %collect data
    loss_train_mat = [loss_train_mat; loss_train];
    acc_train_mat = [acc_train_mat; acc_train];
    loss_test_mat = [loss_test_mat; loss_test];
    acc_test_mat = [acc_test_mat; acc_test];
    
    %save target models
    if acc_test > 0.96
        save('over_96_model.mat','model');
        break
    end
    
end

toc 

figure;
subplot(2,1,1);plot(loss_train_mat);title('TRAIN LOSS');
%axis([0 inf 0 1]);
subplot(2,1,2);plot(acc_train_mat);title('TRAIN ACCURACY');
%axis([0 inf 0 1]);
figure;
subplot(2,1,1);plot(loss_test_mat);title('TEST LOSS');
%axis([0 inf 0 1]);
subplot(2,1,2);plot(acc_test_mat);title('TEST ACCURACY');
%axis([0 inf 0 1]);

