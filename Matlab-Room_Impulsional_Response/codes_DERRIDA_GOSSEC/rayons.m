function [ray] = rayons(N,source,ammortissement_mur,ammortissement_micro) 

    %cette fonction intialise l'objet ray
    x_s = source(1,1);
    y_s = source(1,2); 
    ray.pos = ones(N,2); 
    ray.dir = ones(N,2);
    ray.dist = ones(N,1); %pour garder en mémoire la distance parcourue par le rayon
    ray.last_dist = ones(N,1) ; %pour garder en tête la dernière longueur (car notre programme de mesure ne tient pas compte des murs
    ray.intensity = ones(N,1); %l'intensité vaut 1 au début puis elle sera modifiée au fur et a mesure
    ray.ammortissement_mur = ammortissement_mur; 
    %valeur comprise entre 0 et 1 par lequelle l'intensité est multipliée lorsque le rayon rebondit sur un mur
    ray.ammortissement_micro = ammortissement_micro;
    ray.nb = N;
    
    ray.pos(:,1) = ray.pos(:,1)*x_s ;
    ray.pos(:,2) = ray.pos(:,2)*y_s;
    for k=1:N
        ray.dir(k,1) = cos(2*pi*k/N);
        ray.dir(k,2) = sin(2*pi*k/N);
    end
  
