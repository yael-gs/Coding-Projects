function [energie_recue] = intersection(ray,mic) 

%{ Cette fonction prend les rayons renvoie l'energie recue par le micro. Proportionnelle au nombre de rayons ayant intersecté le micro. %}


    energie_recue = 0;  %l'energie_recue = somme des intensités des rayons recus.
    xm = mic.pos(1,1);
    ym = mic.pos(1,2);
    for k=1:size(ray.dir,1) 
    
        %les calculs suivants ont étés faits sur papier
        a = ray.pos(k,1);
        b = ray.pos(k,2);
        ux = ray.dir(k,1);
        uy = ray.dir(k,2);
        B = 2*uy*(b-ym)+ 2*ux*(a-xm);
        D = (a-xm)^2+ (b-ym)^2 - mic.rad^2;
        det = B^2 - 4*D; 
        % fins de calculs sur papier 
        
        if det>=0 %alors le rayon interesecte le micro
            sol_1 = (-B - sqrt(det)) / 2;
            sol_2 = (-B + sqrt(det))/ 2;
            if sol_1*sol_2 >=0 
                energie_recue = energie_recue + ray.intensity(k); 
                %{ on introduit une notion d'intensité de rayon comprise entre 1 et 0 (pas démandé mais on est zélés) }%
            end
        end
     end
    
            
            


