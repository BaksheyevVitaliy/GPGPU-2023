M = readmatrix("GPGPUExperimentResults.txt");
x = M(:,1);
figure;
loglog(x,M(:,2));
grid on;
hold on;
loglog(x,M(:,3));
loglog(x,M(:,4));
loglog(x,M(:,5));
legend("CPU","GPU non shared","GPU shared row","GPU shared square");
xlabel("N");
ylabel("ms");

figure;
semilogx(x,M(:,3)./M(:,4));
grid on;
hold on;
loglog(x,M(:,3)./M(:,5));
legend("GPU shared row","GPU shared square");
xlabel("N");