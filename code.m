clear all
close all
clc

%Loading the csv file
data=csvread('data.csv');
%setting a seed - to generate random numbers that are repeatable, every
%time the generator is initialized using same seed
rand('seed',1);
%splitting of training and testing data
[a,b] = size(data) ;
split = 2/3 ;
index = randperm(a)  ;
Training = data(index(1:round(split*a)),:) ; 
Testing = data(index(round(split*a)+1:end),:) ;

%Splitting of inputs and outputs from data file
%Input and target training matrices
x=Training(: , 2:10);
t=Training(:,11);

%Input and target testing matrices
x2=Testing(: , 2:10);
t2=Testing(:,11);

x=[x ones(size(x,1),1)];
x2=[x2 ones(size(x2,1),1)];

%trnspose of training input matrix
X=x';

%Constructing training target matrix by replacing 11th attribute with +1's and -1's
y=zeros([466,1]);
for m=1:length(t)
    if t(m)==2
        y(m,:)=[-1];
    else t(m)==4
          y(m,:)=[+1] ;
        
          
    end
end

%Constructing testing target matrix by replacing 11th attribute with +1's and -1's
y2=zeros([233,1]);
for m=1:length(t2)
    if t2(m)==2
        y2(m,:)=[-1];
    else t2(m)==4
          y2(m,:)=[+1] ;
        
          
    end
end

%Number of instances
N=size(x,1);

%Number of dimensions(features)
D=size(x,2);

%weight vector
w=zeros(D,1);

%Aggresiveness parameter
C=input('Enter C value:');

%Number of iterations

Iter=input('Enter number of iterations:');
 
m=0;%to calculate total no. of data loop through the loop depending on no.of iterations
 h=0;%to calculate total no. of misclassified data accross all the iterations
 ptrain=zeros([Iter,1]);
 func = input('Enter a number 1.PA1 2.PA 3.PA2 : ');       
for i = 1:Iter
    m=m+466;
    pred_train=zeros([N,1]);%matrix to get predicted output of train set
  for n = 1:N
  x1 = X(:,n);
  y_pred =sign (dot(w',x1));%predicted output
  pred_train(n,:)=[y_pred];
  
  Y=y(n,1);
  L=(1-(Y*y_pred));%hinge loss
  %lagrang multiplier-optimization
 
 switch func
     case 1
    T= min(C,L/(norm(x1,2).^2 ));%(PA-1)
     case 2
    T=(L/(norm(x1,2).^2));%(PA)
     case 3
   T=(L/(norm(x1,2).^2+1/2*C));%(PA-2)
 end
    if(L>0) %missclassification error
       
        w=w+(T*Y*x1);%weight update
        w;
        h=h+1;
        h;
%         w(n+1)=min(1/2*2*norm(w(n-1)-w(n)).^2)
        %else if (0<L>1)%margin error

            else%no error-loss equals zero
                w=w;
                
            %end
    end
 
errtrain=[(h/m)];
errtrain;%error of train data
%accuracy of training data
acctrain=[(m-h)/m];  
ptrain(i,:)=[errtrain];%saving the train error through all ierations to a matrix
  
           
    
  end   
end

X2=x2';
 N2=size(x2,1);

f=0;%to calculate no. of data loop through the loop depending on no.of iterations
 g=0;%to calculate no. of misclassified data accross all the iterations
 
    f=f+233;
    
    
    pred_test=zeros([N2,1]);%matrix to get the predicted outputs of test set
  for n2 = 1:N2
      
  x3 = X2(:,n2);
  Y_pred = sign(dot(w',x3));%predicted output
  pred_test(n2,:)=[Y_pred];
  
  Y2=y2(n2,1);
    L2=(1-(Y2*Y_pred));%hinge loss
     if(L2>0) %missclassification error
   
        g=g+1;
        g;
    
         errtest=[(g/f)];
errtest;%error of test data
acctest=[(f-g)/f];%accuracy of test data
;%save the testing error through all iterations to a matix
        
     end 
  end

% Plot of errors of training data through all iterations
fig1=figure;
scatter(1:Iter,ptrain,'ro')
xlabel('Iterations');
ylabel('Error');
title('Error vs Iterations of train data');




