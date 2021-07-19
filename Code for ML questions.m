%--------------------- Group 56 ---------------------
%-CH19B012-SUJITH KUMAR------------------------------
%-CH19B052-CH GOVARDHANA SAI SRINIVAS----------------
%-CH19B055-G BHUVAN CHANDRA----------------------------

%------------------Machine Learning------------------

fprintf("---------------Machine Learning-----------\n\n")
%-------------Question 1(Conceptual)-----------------
for i=1:5
 fprintf('-----Q1 V%d (Question 1 , Variant %d)--------\n\n',i,i)
 fprintf("Which of the following statements are true ?\n\n")

 C= randi([2,4]);
 C1=C+1;
 C2=C-1;
 
 ic=randi([2,3]);
 B=ic-1;
 D=ic+1;


 Crct=["It is possible to design a Linear regression algorithm using a neural network."...
    ,"If the size of training data increases , Bias increases and Variance decreases."...
    "For the model with high variance the cost function or squared error function (JӨ) will be low."...
    "Traning neural network has the potential of overfitting the training data"...
    "If the modal bias and variance both are low,the modal will have higher accuracy "];

 Incrct=["Overfitting is more likely when you have huge amount of data to train"...
    ,"Logistic Regression is used for predicting continuous dependent variable"...
    "MLE estimates are often desirable because they have low variance"...
    'Using a model with high bias is always better than using a model with less bias'];

EIC1="False.With a small training dataset, it’s easier to find a hypothesis to fit the training data exactly i.e. overfitting.";
EIC2="False.Logistic Regression is used for classification problems.Hence it is not used for predicting continuous dependent variable.";
EIC3="False.Variance in Maximum likelihood Estimate(MLE) is high.High variance indicated measurement uncertainity hence they are undesirable";
EIC4="False.Bias is the difference between your model's expected predictions and the true values.Low bias algorithms trains model that are accurate on average";

EC1="True.Neural network can be used as a universal approximator, so it can definitely implement a linear regression algorithm.";
EC2="True.As we increase the size of the training data, the bias would increase while the variance would decrease.";
EC3="True.For model with high variance the hypothesis function fits the training data very well which causes the error to be low.";
EC4="True.Overfitting of the training data happens if neural network model is suffering from high variance.'It means the trained parameters fits the training set well,but perform poorly on validation data ";
EC5="True.The statement is self explanatory";

CrctExp=[EC1,EC2,EC3,EC4,EC5];
IncrctExp=[EIC1,EIC2,EIC3,EIC4];

fprintf(['A) %s\n\n'...
         'B) %s \n\n'...
         'C) %s \n\n'...
         'D) %s \n\n'...
         'E)None of these\n\n'],Incrct(B),Crct(C1),Crct(C2),Incrct(D))
fprintf("Answer : B C\nExplanation : \n\n")
fprintf(['A) %s \n\n'...
         'B) %s \n\n'...
         'C) %s \n\n'...
         'D) %s \n\n'],IncrctExp(B),CrctExp(C1),CrctExp(C2),IncrctExp(D))
clear
end

%-------------Question 2(Numerical)-----------------
%--------Can produce any number of variants --------

for i=1:5
fprintf('-----Q2 V%d (Question 2 , Variant %d)-----\n\n',i,i)
p=randi([8,12]);
var=['c','a','b'];
v=randi([1,3]);
step=randi([1,2]);
t=0:step:p;
n=size(t,2);
h=rand(1,n);
fprintf(['The water level in the North sea is mainly determinedby so called M2 tide whose period is about %d hours.\n'...
    'The height H(t) thus roughly taken the form \n\t\t H(t)= c + a sin(2πt/%d) + b cos(2πt/%d)\nUse method of least squares to find %s\n'],p,p,p,var(v))
T = table(t(:),h(:),'VariableNames',{'t(hours)','H(t)(meters)'}); 
disp(T)
sine=sin(2*pi.*t/p);
cosine=cos(2*pi.*t/p);
X=[ones(n,1) sine' cosine'];
thetha= inv((X'*X))*X'*h';
Ans=thetha(v);
options=[Ans+0.25 Ans Ans+0.5 Ans-0.5];
   
    o=randi([1,4]);
    oa=o+1;
    if(oa >4) 
        oa=oa-4;
    end
    ob=o-1;
    if(ob<1) 
        ob=ob+4;
    end
    oc=o;
    od=o-2;
    if(od<1) 
        od=od+4;
    end
   
    if(options(oa) == Ans)
        answer='A';
    elseif(options(ob) == Ans)
        answer='B';
    elseif(options(oc) == Ans)
        answer='C';
    else
        answer='D';
    end
fprintf(['A)%.4f\n'...
         'B)%.4f\n'...
         'C)%.4f\n'...
         'D)%.4f\n'...
         'E)None of these\n'],options(oa),options(ob),options(oc),options(od))
 fprintf('Answer: %s\nSolution :\n',answer)
 fprintf(['Given \n\t H(t)= c + a sin(2πt/%d) + b cos(2πt/%d)\n Using least square method Sr = Σ(hi - (c + a*sin(2πti/%d) + b*cos(2πti/%d)))^2 \n'...
          'To find best possible values of a,b,c equate\n\t del Sr/del a = del Sr/del b =del Sr/del c =0\nOn solving the three equations\n'...
          'we get a= %.4f , b = %.4f , c =%.4f\n\n'],p,p,p,p,thetha(2),thetha(3),thetha(1))
clear
end


%-------------Question 3(Conceptual)-----------------
%----------------Machine Learning----------

for i=1:5
 fprintf('-----Q3 V%d (Question 3 , Variant %d)-----\n\n',i,i)
 fprintf("Which of the following statements are Incorerct about PCA(Principal Component Analysis)?\n\n")

 C= randi([2,4]);
 C1=C+1;
 C2=C-1;
 
 ic=randi([2,3]);
 B=ic-1;
 D=ic+1;


 Crct=["Even if all the input features are on very similar scales, we should still perform mean normalization (so that each feature has zero mean) before running PCA."...
    ,"Given an input x ∈ R^n, PCA compresses it to a lower-dimensional vector z ∈ R^k."...
    "If the input features are on very different scales, it is a good idea to perform feature scaling before applying PCA."...
    "Given input data x ∈ R^n, it makes sense to run PCA only with values of k that satisfy k <= n. where k is the dimension to which the input data reduced"...
    "All principal components are orthogonal to each other and Maximum number of principal components <= number of features"];

 Incrct=[" PCA is susceptible to local optima; trying multiple random initializations may help."...
    ,"when the features reduces to lower dimensions using PCA,the features carries all information present in data"...
    "PCA can be used only to reduce the dimensionality of data by 1 (such as 3D to 2D, or 2D to 1D)."...
    "PCA will perform outstandingly  when eigenvalues are roughly equal"];

EIC1="False.PCA is a deterministic algorithm which doesn’t have local minima problem like most of the machine learning algorithms has.";
EIC2="False.When the features reduces to lower dimensions ,most of the times some information of data will lose and won’t be able to interpret the lower dimension data.";
EIC3="False.PCA can be used to reduce the dimensionality of data to any dimensions less than the given dimension";
EIC4="False. When all eigen vectors are same in such case you won’t be able to select the principal components because in that case all principal components are equal.";


EC1="True.If you do not perform mean normalization, PCA will rotate the data in a possibly undesired way.";
EC2="True.PCA compresses given input to a lower dimensional vector by projecting it onto the learned principal components";
EC3="True.Feature scaling prevents one feature dimension from becoming a strongvprincipal component only because of the large magnitude of the feature values (as opposed to large variance on that dimension).";
EC4="True.With k = n, there is no compression, so PCA has no use and k > n does not make sense.";
EC5="True.The statement is Self explanatory.";

CrctExp=[EC1,EC2,EC3,EC4,EC5];
IncrctExp=[EIC1,EIC2,EIC3,EIC4];

fprintf(['A) %s\n\n'...
         'B) %s \n\n'...
         'C) %s \n\n'...
         'D) %s \n\n'...
         'E)None of these\n\n'],Crct(C1),Incrct(B),Crct(C2),Incrct(D))
fprintf("Answer : A C\nExplanation : \n\n")
fprintf(['A) %s \n\n'...
         'B) %s \n\n'...
         'C) %s \n\n'...
         'D) %s \n\n'],CrctExp(C1),IncrctExp(B),CrctExp(C2),IncrctExp(D))
clear
end

%-------------Question 4(Numerical)-----------------
%--------Can produce any number of variants --------
for i=1:5
  
s=10:30;
s1=1:20;
r=randi([1,20]);
n=s(r)*1000;

n1=s1(r)*1000;
n2=n-n1;

FP=randi([50,n1]);
TN=n1-FP;

TP=randi([50,n2]);
FN=n2-TP;

fprintf('-----Q4 V%d (Question 4 , Variant %d)-----\n\n',i,i)
fprintf(['Suppose %d patients get tested for flu out of them %d are actually healthy and %d are actually sick.\n'...
         'For the sick people a test was positive for %d and negative for %d.For healthy people, the same test was\n'...
         'positive for %d and negative for %d.Calculate the F1 score for the data\n'],n,n1,n2,TP,FN,FP,TN);
     
precision = TP/(TP+FP);
recall= TP/(FN+TP);
F1score= 2*(precision*recall)/(recall+precision);

options=[precision*recall 0.25*precision*recall F1score/4 F1score];
   
    o=randi([1,4]);
    oa=o+1;
    if(oa >4) 
        oa=oa-4;
    end
    ob=o-1;
    if(ob<1) 
        ob=ob+4;
    end
    oc=o;
    od=o-2;
    if(od<1) 
        od=od+4;
    end
   
    if(options(oa) == F1score)
        answer='A';
    elseif(options(ob) == F1score)
        answer='B';
    elseif(options(oc) == F1score)
        answer='C';
    else
        answer='D';
    end
fprintf(['A)%.2f\n'...
         'B)%.2f\n'...
         'C)%.2f\n'...
         'D)%.2f\n'...
         'E)None of these\n'],options(oa),options(ob),options(oc),options(od))
fprintf('Answer: %c \nSolution :\n',answer)
T = table([TP ; FN ],[FP ; TN ],'VariableNames',{'No of Actual sick','No of Actual Healthy '},'RowName',{'No of predicted sick','No of predicted Healthy'}); 
disp(T)
fprintf('\t\tTrue Positives(TP)  : %d\t False Positives(FP) : %d\n\t\tFalse Negatives(FN) : %d\t True Negatives(TN)  : %d\n',TP,FP,FN,TN)
fprintf(['Precision quantifies the number of positive class predictions that actually belong to the positive class.\n'...
         'Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.\n'...
         'F1 score provides a single score that balances both the concerns of precision and recall in one number'])
 fprintf(['Therefore\n\t\tprecision = TP/(TP+FP) = %d/(%d + %d) = %.2f\n'...
          '\t\t\trecall= TP/(TP+FN) = %d/(%d + %d)= %.2f\n'],TP,TP,FP,precision,TP,TP,FN,recall)
 fprintf('\t\t\tF1score= 2*(precision*recall)/(recall+precision)\n\t\t\t\t   = %.2f\n\n',F1score)
 clear
 end
