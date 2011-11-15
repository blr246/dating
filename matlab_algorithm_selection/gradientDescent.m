function [Theta,candidate] = gradientDescent(trainingExamples,labels,alpha,lambda,num_iterations)

[m,n] = size(trainingExamples);
X = trainingExamples;
y = labels;

numAttributes = size(X,2);
Theta = zeros(numAttributes,1);
gradient = zeros(size(Theta));

for i = 1:num_iterations
    J = 1/(2*m) * (Theta' * X' - y').^2;
    gradient = (1/m) .* ((Theta' * X' - y') * X)';
    Theta = Theta * (1 - (alpha * lambda/m)) - (alpha/m) * gradient;
end
Theta = Theta(:);
candidate = double(Theta > 0);
end