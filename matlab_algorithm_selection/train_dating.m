function [results] = train_dating(num_iterations,num_features,num_negweights,numInstances,lambda)
dotprods = zeros(num_iterations,1);
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
initial_theta = zeros(num_features,1);
for i = 1:num_iterations
    [weights,ideal,nonideal,X,y] = getWeights(num_features,num_negweights,numInstances);
    
    %[hyp,c] = gradientDescent(X,y,.01,lambda,1000);
    [hyp,c] = matchMaker(X,y,lambda);
    dotprods = zeros(numInstances,1);
    for examples = 1:numInstances
        dotprods(examples,:) = dot(c,weights);
        X = vertcat(X,hyp');
        y = vertcat(y,dotprods(examples,:));
        %[hyp,c] = gradientDescent(X,y,.01,lambda,1000);
        [hyp,c] = matchMaker(X,y,lambda);
    end
    results(:,i) = dotprods;
    %disp('size dotprods: '); size(dotprods)
    %[theta,J,exit_flag] = fminunc(@(t)(costFunction(t, te, labels, lambda)), initial_theta, options);
    %c = double(theta > 0);
    %dotprods(i,1) = dot(c,weights);
end
%plot(1:num_iterations,results);