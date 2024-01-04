% Load the Iris dataset
load fisheriris
inputs = meas'; % Features (sepal length, sepal width, petal length, petal width)
targets = zeros(3, size(species, 1)); % Create target matrix
for i = 1:size(species, 1)
    if strcmp(species{i}, 'setosa')
        targets(1, i) = 1;
    elseif strcmp(species{i}, 'versicolor')
        targets(2, i) = 1;
    else
        targets(3, i) = 1;
    end
end

% Create and configure the neural network
hiddenLayerSize = 12; % Number of neurons in the hidden layer
net = patternnet(hiddenLayerSize); % Create a pattern recognition network
net.divideParam.trainRatio = 0.9; % Set the training ratio
net.divideParam.valRatio = 0.18; % Set the validation ratio
net.divideParam.testRatio = 0.15; % Set the test ratio

% Train the neural network
[net, tr] = train(net, inputs, targets);

% Test the trained network
outputs = net(inputs);
errors = gsubtract(targets, outputs);
performance = perform(net, targets, outputs);

% Plot confusion matrix
plotconfusion(targets, outputs);

% Display the network
view(net);
