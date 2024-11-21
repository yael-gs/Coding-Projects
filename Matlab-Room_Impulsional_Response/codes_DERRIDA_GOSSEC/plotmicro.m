function plotmicro(mic) % Fonction qui affiche le micro

hold on;
x = mic.pos(1);
y = mic.pos(2);
r = mic.rad;
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit);
hold off