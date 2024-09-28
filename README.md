- 👋 Hi, I’m @sandroem
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
sandroem/sandroem is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
close all; clear all; clc;

dados_treinamento = (importdata('tab_treinamento1.dat'))';
dados_teste = (importdata('tab_teste1.dat'))';

%Criando os vetores de entrada e saída para treinamento e teste da rede
x = dados_treinamento(1:3,:); %vetor com os dados de treinamento (entrada)
t = dados_treinamento(4,:); %vetor com os dados de treinamento (saída)
x_t = dados_teste(1:3,:); %vetor com os dados de teste (entrada)
num_treino = size(x,2); %Qtd amostras de treino
num_teste = size(x_t,2); %Qtd amostras de teste
num_entrada = size(x,1); %Qtd de entradas

% Normalização dos dados de treinamento
[xn, xs] = mapminmax(x);
[tn, ts] = mapminmax(t);
% Normalização dos dados de teste
[x_t_n, x_t_s] = mapminmax(x_t);

% Criação da Rede Perceptron
trainFcn = 'trainlm';  % Levenberg-Marquardt
hiddenLayerSize = 1;  
net = fitnet(hiddenLayerSize, trainFcn);

% Configuração da divisão dos dados para treinamento, validação e teste
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Inicialização dos pesos e bias
net = configure(net, xn, t);
net.IW{1} = rand(hiddenLayerSize, size(xn,1));  % Inicializando os com 
% valores aleatórios 
net.LW{2,1} = 1;
net.b{1} = rand(hiddenLayerSize, 1);  % Inicializando o Bias inicial com 
% valores aleatórios para camada oculta
net.b{2} = 0;


% Exibição dos pesos e Bias iniciais
disp('Pesos Iniciais (Entrada para Camada Oculta):');
disp(net.IW{1});
disp('Pesos Iniciais (Camada Oculta para Saída):');
disp(net.LW{2,1});
disp('Bias Iniciais:');
disp(net.b);

% Treinamento da Rede
[net, tr] = train(net, xn, t);

% Teste da Rede
y = net(xn);

% Exibir os pesos e bias finais
w1 = net.IW{1};
w2 = net.LW{2,1};
b1 = net.b{1};
b2 = net.b{2};

disp('Pesos Finais (Entrada para Camada Oculta):');
disp(w1);
disp('Pesos Finais (Camada Oculta para Saída):');
disp(w2);
disp('Bias Finais:');
disp(b1);
disp(b2);


% Obter as previsões da rede
y_t_n = net(x_t_n);

classe = string(num_teste);
for i=1:num_teste
         
        if y_t_n(i) <= 0
            y1(i) = -1;
            classe(i) = 'C1';
        else
            y1(i) = 1;
            classe(i) = 'C2';
        end;
end

disp(classe);


