function plot_lignes(ray1,ray2) 

%%Cette fonction plot la ligne, rayon par rayon,  entre les centres des deux objets ray entrés 

hold on;

for k =1:ray1.nb 
    x=[ray1.pos(k,1),ray2.pos(k,1)];
    y=[ray1.pos(k,2),ray2.pos(k,2)];
    plot(x,y,'b--'); 
end
hold off
end