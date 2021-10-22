
clc;clear all;close all;

noise = randn(10000,1);
histogram(noise);title('Histogram of Noise signal')
f = fft(noise);
figure;histogram(real(f));title('Histogram of Noise in frequency domain')




figure;
Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 10000;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
X = S + 1*randn(size(t));

plot(1000*t(1:50),S(1:50))
title('Original Signal')
xlabel('t (milliseconds)')
ylabel('S(t)')

figure;
plot(1000*t(1:50),X(1:50))
title('Noisy Signal')
xlabel('t (milliseconds)')
ylabel('X(t)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
Y = fft(X);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of Noisy Signal')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
Y = fft(S);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of Original Signal')
xlabel('f (Hz)')
ylabel('|P1(f)|')


