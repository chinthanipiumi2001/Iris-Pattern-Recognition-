% Step 1: Load the Iris dataset
load fisheriris;

% Select only two features for simplicity
X = meas(:, 1:2);
y = species;

% Step 2: Shuffle the dataset and split into training and testing sets
rng(42); % Set seed for reproducibility
shuffledIndices = randperm(size(X, 1));
trainingIndices = shuffledIndices(1:round(0.6 * length(shuffledIndices)));
testingIndices = shuffledIndices(round(0.6 * length(shuffledIndices)) + 1:end);

% Create training and testing data
X_train = X(trainingIndices, :);
y_train = y(trainingIndices);

X_test = X(testingIndices, :);
y_test = y(testingIndices);

% Step 2.1: Feature scaling
X_train_scaled = zscore(X_train);
X_test_scaled = zscore(X_test);

% Step 3: Evaluate different K values (K=5 and 7) using a loop
k_values = [5, 7];
figure;

for i = 1:length(k_values)
    k = k_values(i);
    
    % Train the KNN classifier
    mdl = fitcknn(X_train_scaled, y_train, 'NumNeighbors', k);

    % Step 4: Report the confusion matrix and percentage of correct classifications
    y_pred = predict(mdl, X_test_scaled);

    % Display confusion matrix
    subplot(2, 3, i);
    C = confusionmat(y_test, y_pred);
    confusionchart(C, unique(y_test), 'Title', sprintf('Confusion Matrix for K=%d', k));

    % Calculate percentage of correct classifications
    accuracy = sum(diag(C)) / sum(C(:));
    fprintf('Accuracy for K=%d: %.2f%%\n\n', k, accuracy * 100);

    % Step 5: Plot the decision boundary (pattern recognition graph)
    subplot(2, 3, i + 3);
    plotDecisionBoundary(mdl, X_test_scaled, y_test, k);
    title(sprintf('Decision Boundary for K=%d', k));
end

% Step 6: Scatter plot of the original dataset
figure;
gscatter(X(:, 1), X(:, 2), y, 'rbg', 'o', 8);
title('Scatter Plot of Original Dataset');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'Best');

% Step 7: Scatter plot of training and testing data
figure;
subplot(1, 2, 1);
gscatter(X_train_scaled(:, 1), X_train_scaled(:, 2), y_train, 'rbg', 'o', 8);
title('Scatter Plot of Training Data');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'Best');

subplot(1, 2, 2);
gscatter(X_test_scaled(:, 1), X_test_scaled(:, 2), y_test, 'rbg', 'o', 8);
title('Scatter Plot of Testing Data');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Class 1', 'Class 2', 'Class 3', 'Location', 'Best');

% Step 8: Limitations or drawbacks of KNN
fprintf('Observations and Analysis:\n');
fprintf('- KNN can be sensitive to the choice of K value.\n');
fprintf('- The algorithm may not perform well on high-dimensional data or when features are not informative.\n');
fprintf('- It is computationally expensive during the testing phase, especially with large datasets.\n');
fprintf('- The decision boundary is influenced by outliers, and the algorithm may not work well with noisy data.\n');
fprintf('- KNN does not provide insights into the underlying relationships in the data.\n');

% Helper function to plot decision boundary
function plotDecisionBoundary(model, X, y, k)
    h = 0.01;
    x1range = min(X(:, 1)):h:max(X(:, 1));
    x2range = min(X(:, 2)):h:max(X(:, 2));
    
    [X1, X2] = meshgrid(x1range, x2range);
    XGrid = [X1(:), X2(:)];
    
    % Predict the classes for each point in the meshgrid
    yGrid = predict(model, XGrid);
    
    % Plot the decision boundary
    gscatter(XGrid(:, 1), XGrid(:, 2), yGrid, 'rbg', '.', 20);
    hold on;
    
    % Plot the training data
    gscatter(X(:, 1), X(:, 2), y, 'k', 'o', 10, 'on');
    
    title(sprintf('Decision Boundary (K=%d)', k));
    xlabel('Feature 1');
    ylabel('Feature 2');
    legend('Class 1', 'Class 2', 'Class 3', 'Training Data', 'Location', 'Best');
    hold off;
end