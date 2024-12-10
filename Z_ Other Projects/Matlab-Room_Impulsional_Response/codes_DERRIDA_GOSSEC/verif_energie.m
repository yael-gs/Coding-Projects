function verif_energie(ray,mic)


N_points = 5000;
max = 500;
min = 3;

X = linspace(min,max,N_points); % Création de l'abscisse du graphe
Y_Eth = zeros(1,N_points);
Y_En = zeros(1,N_points);


for k = 1:N_points
    
    mic = micro(X(1,k),mic.pos(1,2),mic.rad);  % On met à jour le micro à chaque début de boucle
    
    d = sqrt((ray.pos(1,1)-mic.pos(1,1))^2 +(ray.pos(1,2)-mic.pos(1,2))^2);% Calcul de la distance entre le micro et la source sonore
    N_intersection = intersection(ray,mic); 
    
    Y_En(1,k) = N_intersection / (ray.nb*2); % Mesure numérique
    Y_Eth(1,k) = 2*mic.rad/(2*pi*d); % Calcul de l'énergie théorique

    

end
    hold on;

    plot(X,Y_Eth,'r-'); % On trace l'allure expérimentale
    plot(X,Y_En,'k-'); % On trace l'allure théorique
    xlabel('Distance parcourue par le rayon entre le micro et la source') 
    ylabel('Energie')
    title('Comparaison entre l energie théorique et experimentale')
    legend({'y = Eth','y = Eexp'},'Location','northeast')

    hold off;
end