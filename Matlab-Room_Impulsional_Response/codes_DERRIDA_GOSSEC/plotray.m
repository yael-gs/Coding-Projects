function plotray(ray) % Fonction qui affiche les rayons

hold on;

for k =1:ray.nb 
    plot(ray.pos(k,1),ray.pos(k,2),'o'); % On trace les centres des sources
    quiver(ray.pos(k,1),ray.pos(k,2),ray.dir(k,1),ray.dir(k,2),'r'); % On trace les rayons
end
hold off