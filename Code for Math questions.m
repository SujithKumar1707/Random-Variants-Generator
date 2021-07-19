%--------------------- Group 56 ---------------------
%-CH19B012-SUJITH KUMAR------------------------------
%-CH19B052-CH GOVARDHANA SAI SRINIVAS----------------
%-CH19B055-G BHUVAN CHANDRA----------------------------


%-----------PROBABILITY AND STATISTICS---------------
%-------------Question 1(Numerical)-----------------
%--------Can produce any number of variants --------
fprintf('-----------PROBABILITY AND STATISTICS---------------');
for k=1:5
    fprintf('\n----------Q1 V%d (Question 1 , Variant %d)-------------\n\n', k,k);
    number = randi([2,12],1,1);
    numfaces=6;
    step=randi([2,20],1,1);
   
    
    %possibilities when two dies are rolled
    %2-(1,1) 3-(1,2),(2,1) 4-(1,3),(2,2),(3,1)
    %5-(1,4),(2,3),(3,2),(4,1) 6-(1,5)(2,4)(3,3)(4,2)(5,1)
    %7-(1,6)(2,5)(3,4)(4,3)(5,2)(6,1) 8-(2,6)(3,5)(4,4)(5,3)(6,2)
    %9-(3,6)(4,5)(6,3)(5,4) 10-(4,6)(5,5)(6,4)
    %11-(5,6)(6,5) 12-(6,6)
   
    possibilities=containers.Map([1,2,3,4,5,6,7,8,9,10,11,12],[0,1,2,3,4,5,6,5,4,3,2,1]);
    count=possibilities(number);
   
    fprintf(['\nA person is playing a game.Two unbiased dice are thrown at the same time'...
        '\nand the person wins when he gets the sum of two numbers on the dice as %d.'...
        '\nFind the probability of person winning the game in throw %d.\n'], number,step);

    Ans=(((36-count)/36)^(step-1))*(count/36);
    fprintf('\nAns: %.4f\n',Ans);
    fprintf(['\nExplanation: To win he should get a total of %d for the first time in step %d'...
        '\nHence he should not get that total in the previous %d steps'...
        '\nTotal number of possibilities(outcomes) are 36 (as 2 dice)'...
        '\nThe total number of favourable outcomes(i.e getting a sum of %d) are %d'...
        '\nThe probability of not getting that sum in before %d steps is ((36-%d)/36)^(%d)'...
        '\nProbability of getting the required sum at %dth step is (%d/36)'...
        '\nAs both of the above cases should happen,required probability is the product of both the above\n'],number,step,step-1,number,count,step-1,count,step-1,step,count);
clear
end

%-------------Question 2(Numerical)-----------------
%--------Can produce any number of variants --------

for i=1:5
    confidence=[90 95 99];
    Z=[1.645 1.96 2.576];
    n=randi(3);
    zalpha=Z(n);
    sigma=randi(5)^2;
    size=randi([4,100])^2;
    smean=randi([30,500]);
    ll=smean-(zalpha*sqrt(sigma/size));
    Rl=smean+(zalpha*sqrt(sigma/size));

    Al=smean-(sigma);
    AR=smean+(sigma);
    Bl=smean-(sqrt(sigma));
    BR=smean+(sqrt(sigma));
    Dl=smean-0.9*(sigma);
    DR=smean+0.9*(sigma);

    options=[Al,Bl,ll,Dl;AR,BR,Rl,DR];
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
    opt=[oa,ob,oc,od];
    if(options(1,oa) == ll)
        Ans='A';
    elseif(options(1,ob) == ll )
        Ans='B';
    elseif(options(1,oc) == ll)
        Ans='C';
    else
        Ans='D';
    end

    fprintf('-----Q2 V%d (Question 2 , Variant %d)-----\n\n',i,i)
    fprintf('Suppose X has distribution N (µ,σ^2 = %d).A sample of size %d yields sample mean = %d .\nObtain %d %% confidence interval for mean\n\n',sigma,size,smean,confidence(n))
    fprintf(['A)(%.02f %.02f)\n'...
        'B)(%.02f %.02f)\n'...
        'C)(%.02f %.02f)\n'...
        'D)(%.02f %.02f)\n'...
        'E)None of these'],options(1,oa),options(2,oa),options(1,ob),options(2,ob),options(1,oc),options(2,oc),options(1,od),options(2,od))
    fprintf('\nAnswer : %c (%.02f %.02f)\n\n',Ans,ll,Rl)
    fprintf('Solution :\n\n')
    fprintf(['We know that for an confidence interval ɑ P(-z(1-ɑ)/2 <= Y <= z(1-ɑ)/2) has to be ɑ \n'...
       'Where Y is Transformed Normal Variable with mean 0 and variance 1 \n i.e., Y = (Өn - µ)/(σ/n^0.5)\n Өn is sample mean\n'...
       'z(1-ɑ)/2 is z such P(Z<= z) = (1-ɑ)/2\n'...
       'For ɑ = 0.%d z(1-ɑ)/2= %.02f  (From standard normal table)\n'...
       '\tTherefore µ ∈ ( Өn -(z(1-ɑ)/2*(σ/n^0.5),Өn +(z(1-ɑ)/2*(σ/n^0.5)) )\n'...
       '\t⇒ µ ∈ (%.02f-(%.02f *(%d/%d)^0.5) %.02f +(%.02f *(%d/%d)^0.5))\n'...
       '\tTherefore µ ∈ (%.02f %.02f)\n\n'],confidence(n),zalpha,smean,zalpha,sigma,size,smean,zalpha,sigma,size,ll,Rl)
clear
end

%---------------Optimization------------------------
%-------------Question 1(Numerical)-----------------
%--------Can produce any number of variants --------
fprintf('---------------Optimization------------------------\n')

for i=1:5
syms x y alph
val=randi([-10,10],5,1);
c=randi([1,100]);
n=randi(9);
fxy=symfun(val(1)*x^2 + val(2)*x*y + val(3)*y^2+ val(4)*x+ val(5)*y+ c,[x,y]);
[FX]=gradient(fxy,x);
[FY]=gradient(fxy,y);
Xo=randi([-2,2],2,1);
xo=Xo(1);
yo=Xo(2);
try
for j=1:n
  
   Xk=symfun(Xo(1)-alph*FX(Xo(1),Xo(2)),alph);
   Yk=symfun(Xo(2)-alph*FY(Xo(1),Xo(2)),alph);
   falpha= fxy(Xk,Yk);
   grad=gradient(falpha,alph);
   
       alpha1 = solve(grad== 0);
       Xo(1)=Xk(alpha1);
       Xo(2)=Yk(alpha1);
end
Ans=alpha1;
syms alpha
X=symfun(xo- alpha*FX(xo,yo),(alpha));
Y=symfun(yo- alpha*FY(xo,yo),(alpha));
grad0=gradient(fxy(X,Y),alpha);
alphao=solve(grad0==0);
catch 
    i=i-1;
    clear
    continue
end

  options=[Ans+0.5 Ans Ans-0.5 Ans+0.25];
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
    elseif(options(ob) == Ans )
        answer='B';
    elseif(options(oc) == Ans)
        answer='C';
    else
        answer='D';
    end
fprintf('-----Q3 V%d (Question 3 , Variant %d)-----\n\n',i,i)
fprintf(['Find the best possible value for learning rate in gradient descent for the'...
    '\nfunction f(x,y)= %-d*x^2 %+-d*x*y %+-d*y^2 %+-d*x %+-d*y %+-d in %dth step For (x0,y0)=(%d,%d)\n'...
    ],val(1),val(2), val(3),val(4),val(5),c,n,Xo(1),Xo(2))

fprintf(['A)%.2f\n'...
         'B)%.2f\n'...
         'C)%.2f\n'...
         'D)%.2f\n'...
         'E)None of these\n'],options(oa),options(ob),options(oc),options(od))
fprintf('Answer : %c %.2f\nSolution :\n',answer,Ans)

fprintf(['f(x,y)= %-d*x^2 %+-d*x*y %+-d*y^2 %+-d*x %+-d*y %+-d \n'...
         'Initial guess (xo,yo)=(%d,%d)\n'...
         '▽f = [%s ; %s]\n'...
         'Gradient Descent : \n At Iteration k Xk+1 = Xk - ɑ*▽f|(xk,yk)\n'...
         'Iteration 1:\n X1 =Xo- ɑ*▽f|(xo,yo)\n▽f|(xo,yo)= [%d ; %d]\n'...
         'X1=[%d ; %d]-ɑ*[%d ; %d]\n\t ⇒ X1= [%s ; %s]\n'...
         'f(x1,y1)= g(alpha) = %s\n'...
         ],val(1),val(2), val(3),val(4),val(5),c,xo,yo,FX,FY,FX(xo,yo),FY(xo,yo),xo,yo,FX(xo,yo),FY(xo,yo),X(alpha),Y(alpha),fxy(X,Y))
 fprintf("g'(alpha) = %s\n \t⇒alpha= %.2f\n",grad0,alphao)  
if(n>1)
     fprintf(".......After %d iteration optimum Value of ɑ = %.2f \n\n",n,Ans)       
end
clear
end

for h=1:5
fprintf('\n---------Q4 V%d (Question 4 variant %d)---------\n',h,h);
syms x y
lb=randi([5,8]);
sb=randi([1,4]);
p=randi([1,5]);
profitl=90:10:130;
profits=20:10:70;
plb=profitl(p);
slb=profits(p);
n=randi([2,3]);
t=randi([24,36]);
fprintf(['A vendor makes two types of beer cases large and small cases.'...
         '\nHe takes %d hours to make a large case and %d hours to make a short case.'...
         '\nProfit on large case is Rs%d and on small case is Rs%d.He can spend only upto %d hours per week.'...
         '\nHe wishes to make atleast %d cases of each size per week. Find the number of large and small cases'...
         '\nhe should make to maximize his revenue in a week and also Find his profit in that case\n'],lb,sb,plb,slb,t,n);

A=[-1 0;0 -1;lb sb];
b=[-n;-n;t];

f= [-plb ; -slb];

X=linprog(f,A,b);

[n1,n2]=size(X);
if(n1 == 0)
    fprintf(['Ans: There is no optimum solution with these conditions.'...
        '\nTo get a maximized revenue, he needs to modify his work conditions']);
end
if(n1 ~=0)
    if (X(1)-floor(X(1))~=0|| X(2)-floor(X(2)) ~=0)
        %In this case, either number of large or small beercases are not integers. To get a maximum revenue in this situation he should
        % manufacture the beercases to the greatest integer which is less than the obtained value so that his work hours wouldn't exceed his
        %limit ( Example: If x turns out to be 2.4 or 2.6, he should manufacture only 2 but not 3 inorder to get best profit within his work hours
            X(1)=floor(X(1));
            X(2)=floor(X(2));
            total_profit=X(1)*plb+X(2)*slb;
            fprintf(['Ans:Profit is %d'...
                '\nNumber of large cases are %d and small cases are %d'],total_profit,X(1),X(2));
            fprintf(['\nExplanation: Writing the data in the form of equations we get'...
                '\nObjective function is %dx + %dy where x are number of large cases and y are small cases'...
                '\nGiven he can spend upto %d hours per week. Hence %dx+%dy<=%d'...
                '\nHe makes atleast %d of each cases. Hence x>=%d; y>=%d'...
                '\nSolving all the above equations, x and y that satisfies all of them are %d (large cases) and %d (small cases)'...
                '\n Hence Profit is %d*%d+%d*%d=%d\n'],plb,slb,t,lb,sb,t,n,n,n,X(1,1),X(2,1),plb,X(1,1),slb,X(2,1),total_profit);
       
    else
        total_profit=X(1)*plb+X(2)*slb;
            fprintf(['Ans:Profit is %d' ...
                '\nNumber of large cases are %d and small cases are %d'],total_profit,X(1),X(2));
            fprintf(['\nExplanation: Writing the data in the form of equations we get'...
                '\nObjective function is %dx + %dy where x are number of large cases and y are small cases'...
                '\nGiven he can spend upto %d hours per week. Hence %dx+%dy<=%d'...
                '\nHe makes atleast %d of each cases. Hence x>=%d; y>=%d'...
                '\nSolving all the above equations, x and y that satisfies all of them are %d (large cases) and %d (small cases)'...
                '\n Hence Profit is %d*%d+%d*%d=%d\n'],plb,slb,t,lb,sb,t,n,n,n,X(1,1),X(2,1),plb,X(1,1),slb,X(2,1),total_profit);
    end
end
clear
end

%----------------------Linear Algebra--------------------

fprintf("------------------Linear Algebra------------------\n\n")

for i=1:5
 fprintf('-----Q5 V%d (Question 5 , Variant %d)-----\n\n',i,i)
 fprintf("Which of the following statement is True? Explain\n Note : A' is A transpose \n A^-1 is A inverse \n A^n is A*A*....n times\n\n")
 c= randi(3);
 ic=randi([2,5]);
 B=ic-1;
 if(B>5)
     B=B-5;
 end
 C=ic+1;
 if(C>5)
     C=C-5;
 end
 D=ic+2;
 if(D>5) 
     D=D-5;
 end

 Crct=["If the characteristic polynomial of an n×n matrix A is p(λ)=(λ−1)n+2,then A is invertible"...
    ,"If A^2 is an invertible n×n matrix, then A^3 is also invertible."...
    "For every m × n matrix A, the sum of the dimensions of Null space of A' and Column space of A is equal to m."];

 Incrct=["If each entry of an n×n matrix A is a real number, then the eigenvalues of A are all real numbers."...
    ,"If A is a 3×3 matrix such that det(A)=7, then det(2A'* A^−1)=2 "...
    sprintf("If v is an eigenvector of an n×n matrix A with corresponding eigenvalue λ1,and if w is an eigenvector of A with corresponding eigenvalue λ2,\nthen v+w is an eigenvector of A with corresponding eigenvalue λ1+λ2.")...
    "If V is a 6-dimensional vector space, and v1, v2 . . , v6 are six vectors in V ,then they must form a basis of V ."...
    sprintf("A real n×n matrix A = {aij} is defined as follws\n\taij= i if i=j\n\t\t=0 otherwise\nThe summation of all n eigen values of A is n*(n-1)")];

EIC1="False.In general, a real matrix can have a complex number eigenvalue";
EIC2=sprintf("False.We have det(A')=det(A).\n\t⇒ det(2A'A^−1)=det(2I)det(A')det(A^−1)\n\t\t\t\t   =det(2I)=8 \t(since det(A)det(A)^−1=I).\n\t Here I is the 3×3 identity matrix.");
EIC3=sprintf("False.By contradiction:\n\t\t\t If v1+v2 is an eigenvector of A then there exists and eigenvalue λ so that\n\t\t\t A(v1+v2)=λ(v1+v2)=λv1+λv2.\n\t\t\t However since v1 and v2 are eigenvectors and A is linear we have\n\t\t\t A(v1+v2)=A(v1)+A(v2)=λ1v1+λ2v2.\n\t\t\t Therefore\n\t\t\t λv1+λv2=λ1v1+λ2v2⇒\t(λ−λ1)v1+(λ−λ2)v2=0.\n\t\t\t Since λ1≠λ2, v1 and v2 are linearly independent so λ−λ1=0λ−λ2=0.\n\t\t\t So λ=λ1=λ2 which is a contradiction.\n");
EIC4="False. To form basis of v1,v2,...v6 should be linearly independent and span the vector space which may not be possible always";
EIC5=sprintf("False.A is a diagonal matrix containing elements 1,2......n\nAs diagonal elements are eigen values\n\tΣλi=1+2+.....+n=n*(n+1)/2");
EC1=sprintf("True.We have p(0)=(−1)n+2≠0\n\t Thus 0 is not an eigenvalue of A.\n\t A matrix is invertible if and only if it does not have 0 as an eigenvalue.\n\t Thus, A is invertible.\n");
EC2=sprintf("True.If A^2 is invertible,then we have det(A^2)≠0.\n\t Then we have det(A)^2=det(A^2)≠0,and hence det(A)≠0.\n\t It follows that we have det(A^3)=det(A)^3≠0.\n\t Thus the matrix A3 is invertible.\n");
EC3=sprintf("True. Null space of A' is the left null space of A.\nTherefore\tDimensions of Column Space is r\t\twhere r is the rank of matrix\n\t\t\tDimension of Leftnull space is m-r \tand m is the number of rows\n\t sum of dimensions of Null space of A' and Column space of A = m-r+r = m\n");

CrctExp=[EC1,EC2,EC3];
IncrctExp=[EIC1,EIC2,EIC3,EIC4,EIC5];
options=[Crct(c) Incrct(B) Incrct(C) Incrct(D)];
exp=[CrctExp(c) IncrctExp(B) IncrctExp(C) IncrctExp(D)];

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

if(options(oa) == Crct(c))
    Ans='A';
elseif(options(ob) == Crct(c))
    Ans='B';
elseif(options(oc) == Crct(c))
    Ans='C';
else
    Ans='D';
end
fprintf(['A) %s\n'...
         'B) %s \n'...
         'C) %s \n'...
         'D) %s \n'...
         'E)None of These\n'],options(oa),options(ob),options(oc),options(od))
fprintf("Answer : %c\nExplanation : \n",Ans)
fprintf(['A) %s \n'...
         'B) %s \n'...
         'C) %s \n'...
         'D) %s \n\n'],exp(oa),exp(ob),exp(oc),exp(od))
clear
end

for j=1:5
    v1=randi([-10,10],2,1);
    v2=randi([-10,10],2,1);
    v=[v1,v2];
    r=rank(v);
    if r<2
        v1=v1+randi([-15,15],2,1);
    end
    X=randi([-10,10],2,1);
    Xbar=v*inv(v'*v)*v'*X;
    %finding the hyperplane
    b=randi([-15,15],1,1);
    syms f(x1,x2);
    f(x1,x2)= x1*Xbar(1)+x2*Xbar(2)+b;
    point=randi([-10,10],1,2);
    comp=f(point(1,1),point(1,2));
    fprintf("-----Q6 V%d (Question 6 , Variant %d)-----\n\n",j,j);
    fprintf(['Find in which half space does the point (%d,%d) lies corresponding to the hyperplane'...
         '\nwhose n is given by projection of vector [%d;%d] on the'...
         '\nplane spanned by the vectors v1 [%d;%d] and v2 [%d;%d] and  b is %d'...
         '\nA)Positive halfspace'...
         '\nB)Negative halfspace'...
         '\nC)Lies on hyperplane itself\n'],point(1,1),point(1,2),X(1,1),X(2,1),v1(1,1),v1(2,1),v2(1,1),v2(2,1),b);
    if comp>0
        disp("Ans:A");
        fprintf(['Explanation: Projection of X on plane spanned by v1 and v2 is given by Xbar=v*inv(vT*v)*vT*X i.e n=[%.2f;%.2f]'...
            '\n        Equation of hyperplane is xT*n+b=0'...
            '\n        Evaluating xT*n+b at the given point we get %d'...
            '\n        As it is >0, the point lies in positive halfspace'...
            '\n        Note: aT denotes transpose of a\n'],Xbar(1,1),Xbar(2,1),comp);
    elseif comp<0
        disp('Ans:B');
        fprintf(['Explanation: Projection of X on plane spanned by v1 and v2 is given by Xbar=v*inv(vT*v)*vT*X i.e n=[%.2f;%.2f]'...
            '\n        Equation of hyperplane is xT*n+b=0'...
            '\n        Evaluating xT*n+b at the given point we get %d'...
            '\n        As it is <0, the point lies in negative halfspace'...
            '\n        Note: aT denotes transpose of a\n'],Xbar(1,1),Xbar(2,1),comp);
    else
        disp('Ans:C');
        fprintf(['Explanation: Projection of X on plane spanned by v1 and v2 is given by Xbar=v*inv(vT*v)*vT*X i.e n=[%.2f;%.2f]'...
            '\n        Equation of hyperplane is xT*n+b=0'...
            '\n        Evaluating xT*n+b at the given point we get %d'...
            '\n        As it is exactly 0, the point lies on the hyperplane itself'...
            '\n        Note: aT denotes transpose of a\n'],Xbar(1,1),Xbar(2,1),comp);
        
    end
clear    
end